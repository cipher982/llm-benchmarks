import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import dotenv
import typer
from pymongo import MongoClient, ReturnDocument

from llm_bench.config import CloudConfig
from llm_bench.logging import log_mongo
from llm_bench.models_db import load_provider_models
from llm_bench.utils import has_recent_cloud_run


app = typer.Typer(add_completion=False)


# Hardcoded defaults consistent with existing runner
QUERY_TEXT = "Tell a long and happy story about the history of the world."
MAX_TOKENS = 64
TEMPERATURE = 0.1


# Map provider name -> module import path
PROVIDER_MODULES: Dict[str, str] = {
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


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "t", "yes", "y")


def _load_provider_func(provider: str):
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unsupported provider: {provider}")
    module_name = PROVIDER_MODULES[provider]
    module = __import__(module_name, fromlist=["generate"])  # type: ignore
    return module.generate


def _mongo() -> Tuple[str, str]:
    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DB")
    if not uri or not db:
        raise RuntimeError("MONGODB_URI and MONGODB_DB must be set")
    return uri, db


def _errors_collection_name() -> str:
    return os.getenv("MONGODB_COLLECTION_ERRORS", "errors_cloud")


def _jobs_collection_name() -> Optional[str]:
    # Optional; default to 'jobs'. Allow empty to disable.
    return os.getenv("MONGODB_COLLECTION_JOBS", "jobs")


def _log_error_mongo(provider: str, model: str, stage: str, message: str, tb: Optional[str] = None) -> None:
    try:
        uri, db = _mongo()
        coll = _errors_collection_name()
        client = MongoClient(uri)
        try:
            doc = {
                "provider": provider,
                "model_name": model,
                "ts": datetime.now(timezone.utc),
                "stage": stage,
                "message": message,
            }
            if tb:
                doc["traceback"] = tb
            client[db][coll].insert_one(doc)
        finally:
            client.close()
    except Exception:
        # As a last resort, print to console; no file logging in headless path
        print(f"❌ Error {provider}:{model} - {stage} - {message}")


def _log_success_mongo(config: CloudConfig, metrics: Dict) -> None:
    # Write success metrics to existing metrics collection referenced by env
    uri, db = _mongo()
    coll = os.getenv("MONGODB_COLLECTION_CLOUD")
    if not coll:
        raise RuntimeError("MONGODB_COLLECTION_CLOUD not set")
    log_mongo(
        model_type="cloud",
        config=config,
        metrics=metrics,
        uri=uri,
        db_name=db,
        collection_name=coll,
    )


def _claim_next_job() -> Optional[Dict]:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return None
    try:
        uri, db = _mongo()
    except RuntimeError:
        return None
    client = MongoClient(uri)
    try:
        jobs = client[db][coll_name]
        job = jobs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running", "started_at": datetime.now(timezone.utc)}},
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        return job
    finally:
        client.close()


def _complete_job(job_id) -> None:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return
    uri, db = _mongo()
    client = MongoClient(uri)
    try:
        client[db][coll_name].update_one({"_id": job_id}, {"$set": {"status": "done", "finished_at": datetime.now(timezone.utc)}})
    finally:
        client.close()


def _fail_job(job_id, message: str) -> None:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return
    uri, db = _mongo()
    client = MongoClient(uri)
    try:
        client[db][coll_name].update_one(
            {"_id": job_id},
            {"$set": {"status": "error", "error": message, "finished_at": datetime.now(timezone.utc)}}
        )
    finally:
        client.close()


def _validate_metrics(metrics: Dict, max_tokens: int) -> Tuple[bool, Optional[str]]:
    if not isinstance(metrics, dict):
        return False, "metrics not a dict"
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for k in required:
        if k not in metrics:
            return False, f"missing metric: {k}"
    if metrics["tokens_per_second"] <= 0:
        return False, "tokens_per_second <= 0"
    out = metrics["output_tokens"]
    # Some providers may not produce exactly max_tokens; allow ±10%
    if not isinstance(out, int):
        return False, "output_tokens not int"
    if abs(out - max_tokens) > max_tokens * 0.1:
        return False, f"output_tokens {out} not within 10% of requested {max_tokens}"
    return True, None


def _run_single_model(
    provider: str,
    model: str,
    run_always: bool,
    fresh_minutes: int,
    debug: bool,
) -> Dict[str, Optional[str]]:
    # Build configs
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_config = CloudConfig(
        provider=provider,
        model_name=model,
        run_ts=run_ts,
        temperature=TEMPERATURE,
        misc={},
    )

    run_config = {
        "query": QUERY_TEXT,
        "max_tokens": MAX_TOKENS,
    }

    # Skip based on freshness window using Mongo metrics
    if not run_always and fresh_minutes > 0:
        since = datetime.now(timezone.utc) - timedelta(minutes=fresh_minutes)
        if has_recent_cloud_run(model_name=model, provider=provider, since_utc=since):
            print(f"⏭️ Skipped {provider}:{model} (fresh < {fresh_minutes}m)")
            return {"status": "skipped", "reason": f"fresh < {fresh_minutes}m"}

    # Load provider
    generate = _load_provider_func(provider)

    # Execute and log
    try:
        metrics = generate(model_config, run_config)
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)}"
        _log_error_mongo(provider, model, stage="generate", message=reason, tb=traceback.format_exc(limit=5))
        print(f"❌ Error {provider}:{model} - {reason}")
        return {"status": "error", "reason": reason}

    if not metrics:
        msg = "empty metrics"
        _log_error_mongo(provider, model, stage="validate", message=msg)
        print(f"❌ Error {provider}:{model} - {msg}")
        return {"status": "error", "reason": msg}

    ok, why = _validate_metrics(metrics, MAX_TOKENS)
    if not ok:
        _log_error_mongo(provider, model, stage="validate", message=str(why))
        print(f"❌ Error {provider}:{model} - {why}")
        return {"status": "error", "reason": why}

    # Write success only to Mongo metrics collection
    try:
        _log_success_mongo(model_config, metrics)
    except Exception as e:
        reason = f"log failure: {type(e).__name__}: {e}"
        _log_error_mongo(provider, model, stage="log", message=reason)
        print(f"❌ Error {provider}:{model} - {reason}")
        return {"status": "error", "reason": reason}

    # Single concise line per model
    print(
        f"✅ Success {provider}:{model} (tps={metrics.get('tokens_per_second'):.2f}, out={metrics.get('output_tokens')})"
    )

    return {"status": "success"}


@app.command()
def main(
    providers: Optional[str] = typer.Option(
        None,
        "--providers",
        help="Comma-separated list of providers, or 'all' for all available",
    ),
    limit: int = typer.Option(100, "--limit", help="Limit per-provider models"),
    run_always: bool = typer.Option(False, "--run-always", is_flag=True, help="Ignore freshness window for periodic runs"),
    fresh_minutes: int = typer.Option(30, "--fresh-minutes", help="Skip runs newer than this window (minutes)"),
    debug: bool = typer.Option(False, "--debug", is_flag=True, help="Verbose per-model output (unused; always concise)"),
) -> None:
    """Headless runner that calls provider modules directly and writes to Mongo.

    Behavior:
    - Drains ad-hoc jobs (if any) before periodic runs.
    - Skips recent runs within --fresh-minutes unless --run-always or job.ignore_freshness.
    - Prints one concise line per model.
    """
    # Load environment for local/dev; docker-compose also provides env_file
    dotenv.load_dotenv(".env")

    # Load configured models by provider
    provider_models: Dict[str, List[str]] = load_provider_models()

    # Pick providers
    if not providers or providers.strip().lower() == "all":
        selected = list(provider_models.keys())
    else:
        selected = [p.strip() for p in providers.split(",")]

    print(f"Running providers: {selected}")

    # 1) Drain ad-hoc jobs first (optional)
    while True:
        job = _claim_next_job()
        if not job:
            break
        job_provider = job.get("provider")
        job_model = job.get("model") or job.get("model_id") or job.get("model_name")
        ignore_freshness = bool(job.get("ignore_freshness", False))
        if not job_provider or not job_model:
            msg = "invalid job doc (missing provider/model)"
            print(f"❌ Error job - {msg}")
            _fail_job(job.get("_id"), msg)
            continue
        res = _run_single_model(job_provider, job_model, run_always or ignore_freshness, fresh_minutes, debug)
        if res.get("status") == "success":
            _complete_job(job.get("_id"))
        else:
            _fail_job(job.get("_id"), res.get("reason") or "unknown error")

    # 2) Periodic pass for all enabled models per provider
    for provider in selected:
        models = provider_models.get(provider, [])[:limit]
        for model in models:
            _run_single_model(provider, model, run_always, fresh_minutes, debug)


if __name__ == "__main__":
    app()
