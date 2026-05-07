from __future__ import annotations

import os
import signal
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from multiprocessing import Queue
from multiprocessing import get_context
from typing import Any

import dotenv
from pymongo import MongoClient

from llm_bench.config import CloudConfig
from llm_bench.logging import log_mongo
from llm_bench.ops.error_rollups import upsert_error_rollup
from llm_bench.ops.error_taxonomy import classify_error
from llm_bench.scheduler.mongo import error_rollups_collection_name
from llm_bench.scheduler.mongo import errors_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.utils import get_current_timestamp

dotenv.load_dotenv(".env")

QUERY_TEXT = "Tell a long and happy story about the history of the world."
MAX_TOKENS = 64
TEMPERATURE = 0.1

PROVIDER_MODULES: dict[str, str] = {
    "openai": "llm_bench.cloud.providers.openai",
    "anthropic": "llm_bench.cloud.providers.anthropic",
    "bedrock": "llm_bench.cloud.providers.bedrock",
    "vertex": "llm_bench.cloud.providers.vertex",
    "anyscale": "llm_bench.cloud.providers.anyscale",
    "together": "llm_bench.cloud.providers.together",
    "openrouter": "llm_bench.cloud.providers.openrouter",
    "azure": "llm_bench.cloud.providers.azure",
    "runpod": "llm_bench.cloud.providers.runpod",
    "fireworks": "llm_bench.cloud.providers.fireworks",
    "deepinfra": "llm_bench.cloud.providers.deepinfra",
    "groq": "llm_bench.cloud.providers.groq",
    "databricks": "llm_bench.cloud.providers.databricks",
    "lambda": "llm_bench.cloud.providers.lambda",
    "cerebras": "llm_bench.cloud.providers.cerebras",
}

VARIABLE_OUTPUT_PROVIDERS = ("openai", "openrouter", "deepinfra", "fireworks", "together", "groq", "vertex")

_PROVIDER_MODULES_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class RunnerResult:
    status: str
    error_kind: str | None = None
    error_message: str | None = None


def load_provider_func(provider: str):
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unsupported provider: {provider}")
    if provider in _PROVIDER_MODULES_CACHE:
        return _PROVIDER_MODULES_CACHE[provider]
    module = __import__(PROVIDER_MODULES[provider], fromlist=["generate"])  # type: ignore
    generate_func = module.generate
    _PROVIDER_MODULES_CACHE[provider] = generate_func
    return generate_func


def validate_metrics(provider: str, metrics: dict[str, Any], max_tokens: int) -> tuple[bool, str | None]:
    if not isinstance(metrics, dict):
        return False, "metrics not a dict"
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for key in required:
        if key not in metrics:
            return False, f"missing metric: {key}"
    if metrics["tokens_per_second"] <= 0:
        return False, "tokens_per_second <= 0"
    out = metrics["output_tokens"]
    if not isinstance(out, int):
        return False, "output_tokens not int"
    if metrics.get("visible_text_empty") is True:
        if metrics.get("response_status") == "incomplete" or metrics.get("finish_reason") in (
            "length",
            "max_output_tokens",
        ):
            return False, "visible output empty after token budget was exhausted; retry with a larger output budget"
        return False, "visible output text is empty"
    visible_out = metrics.get("visible_output_tokens")
    if visible_out is not None and visible_out <= 0:
        return False, f"visible_output_tokens {visible_out} <= 0"

    if provider in VARIABLE_OUTPUT_PROVIDERS:
        if out <= 0:
            return False, f"output_tokens {out} <= 0 for {provider}"
        return True, None

    if abs(out - max_tokens) > max_tokens * 0.1:
        return False, f"output_tokens {out} not within 10% of requested {max_tokens}"
    return True, None


def validation_policy(provider: str) -> str:
    if provider in VARIABLE_OUTPUT_PROVIDERS:
        return "visible_nonzero"
    return "strict_pm10"


def log_error_mongo(
    *,
    provider: str,
    model_id: str,
    stage: str,
    message: str,
    tb: str | None = None,
    exc_type: str = "",
    client: MongoClient | None = None,
    db_name: str | None = None,
) -> str:
    uri, default_db = mongo_env()
    db_name = db_name or default_db
    should_close = False
    if client is None:
        client = MongoClient(uri)
        should_close = True
    try:
        classified = classify_error(message=message, exc_type=exc_type)
        fingerprint = classified.fingerprint(provider=provider, model=model_id, stage=stage)
        doc = {
            "provider": provider,
            "model_name": model_id,
            "ts": datetime.now(timezone.utc),
            "stage": stage,
            "message": message,
            "error_kind": classified.kind.value,
            "http_status": classified.http_status,
            "provider_error_code": classified.provider_error_code,
            "normalized_message": classified.normalized_message,
            "fingerprint": fingerprint,
        }
        if tb:
            doc["traceback"] = tb
        client[db_name][errors_collection_name()].insert_one(doc)
        upsert_error_rollup(
            collection=client[db_name][error_rollups_collection_name()],
            fingerprint=fingerprint,
            provider=provider,
            model_name=model_id,
            stage=stage,
            error_kind=classified.kind.value,
            normalized_message=classified.normalized_message,
        )
        return classified.kind.value
    finally:
        if should_close:
            client.close()


def log_success_mongo(config: CloudConfig, metrics: dict[str, Any]) -> None:
    uri, db_name = mongo_env()
    log_mongo(
        model_type="cloud",
        config=config,
        metrics=metrics,
        uri=uri,
        db_name=db_name,
        collection_name=metrics_collection_name(),
    )


def run_benchmark_job(job: dict[str, Any]) -> RunnerResult:
    provider = str(job["provider"])
    model_id = str(job["model_id"])
    if job.get("job_kind") == "smoke_hang":
        smoke_seconds = job.get("smoke_seconds")
        seconds = 300 if smoke_seconds is None else int(smoke_seconds)
        print(f"Smoke hang job sleeping for {seconds}s: {provider}:{model_id}", flush=True)
        time.sleep(seconds)
        return RunnerResult(status="success")

    run_ts = get_current_timestamp()
    model_config = CloudConfig(
        provider=provider,
        model_name=model_id,
        run_ts=run_ts,
        temperature=TEMPERATURE,
        misc={},
    )
    run_config = {
        "query": QUERY_TEXT,
        "max_tokens": MAX_TOKENS,
    }

    try:
        generate = load_provider_func(provider)
        metrics = generate(model_config, run_config)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        error_kind = log_error_mongo(
            provider=provider,
            model_id=model_id,
            stage="generate",
            message=reason,
            tb=traceback.format_exc(limit=5),
            exc_type=type(exc).__name__,
        )
        print(f"Error {provider}:{model_id} - {reason}", flush=True)
        return RunnerResult(status="error", error_kind=error_kind, error_message=reason)

    if not metrics:
        msg = "empty metrics"
        error_kind = log_error_mongo(provider=provider, model_id=model_id, stage="validate", message=msg)
        print(f"Error {provider}:{model_id} - {msg}", flush=True)
        return RunnerResult(status="error", error_kind=error_kind, error_message=msg)

    ok, why = validate_metrics(provider, metrics, MAX_TOKENS)
    if not ok:
        error_kind = log_error_mongo(provider=provider, model_id=model_id, stage="validate", message=str(why))
        print(f"Error {provider}:{model_id} - {why}", flush=True)
        return RunnerResult(status="error", error_kind=error_kind, error_message=str(why))

    metrics.setdefault("validation_policy", validation_policy(provider))
    try:
        log_success_mongo(model_config, metrics)
    except Exception as exc:
        reason = f"log failure: {type(exc).__name__}: {exc}"
        error_kind = log_error_mongo(
            provider=provider,
            model_id=model_id,
            stage="log",
            message=reason,
            exc_type=type(exc).__name__,
        )
        print(f"Error {provider}:{model_id} - {reason}", flush=True)
        return RunnerResult(status="error", error_kind=error_kind, error_message=reason)

    print(
        f"Success {provider}:{model_id} "
        f"(tps={metrics.get('tokens_per_second'):.2f}, out={metrics.get('output_tokens')})",
        flush=True,
    )
    return RunnerResult(status="success")


def _child_main(job: dict[str, Any], result_queue: Queue) -> None:
    try:
        if hasattr(os, "setsid"):
            os.setsid()
    except Exception:
        pass
    try:
        result_queue.put(run_benchmark_job(job))
    except BaseException as exc:
        result_queue.put(
            RunnerResult(
                status="error",
                error_kind="unknown",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )
        raise


def _kill_process_group(process) -> None:
    pid = process.pid
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.terminate()
        except Exception:
            return
    process.join(timeout=5)
    if process.is_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            process.kill()


def run_job_in_child(job: dict[str, Any], *, deadline_seconds: int) -> RunnerResult:
    ctx = get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_main, args=(job, result_queue))
    process.start()
    process.join(timeout=deadline_seconds)
    if process.is_alive():
        _kill_process_group(process)
        process.join(timeout=5)
        provider = str(job.get("provider"))
        model_id = str(job.get("model_id"))
        message = f"benchmark timed out after {deadline_seconds}s"
        try:
            log_error_mongo(
                provider=provider,
                model_id=model_id,
                stage="timeout",
                message=message,
                exc_type="TimeoutError",
            )
        except Exception as exc:
            print(f"Failed to log timeout for {provider}:{model_id}: {exc}", flush=True)
        return RunnerResult(status="timeout", error_kind="timeout", error_message=message)

    if process.exitcode and process.exitcode != 0:
        return RunnerResult(status="error", error_kind="unknown", error_message=f"child exited {process.exitcode}")

    if result_queue.empty():
        return RunnerResult(status="error", error_kind="unknown", error_message="child exited without result")

    result = result_queue.get()
    if isinstance(result, RunnerResult):
        return result
    if isinstance(result, dict):
        return RunnerResult(**result)
    return RunnerResult(status="error", error_kind="unknown", error_message=f"invalid child result: {result!r}")
