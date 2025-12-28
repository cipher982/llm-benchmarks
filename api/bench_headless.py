import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import typer
from pymongo import MongoClient, ReturnDocument

from llm_bench.config import CloudConfig
from llm_bench.logging import log_mongo
from llm_bench.models_db import load_provider_models
from llm_bench.ops.error_rollups import upsert_error_rollup
from llm_bench.ops.error_taxonomy import classify_error
from llm_bench.utils import has_recent_cloud_run
from llm_bench.utils import get_current_timestamp

# Load environment for local/dev; docker-compose also provides env_file
dotenv.load_dotenv(".env")

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


# Cache for imported provider modules
_PROVIDER_MODULES_CACHE: Dict[str, Any] = {}


def _load_provider_func(provider: str):
    """Load and cache provider module to avoid repeated imports."""
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unsupported provider: {provider}")

    # Return cached function if already loaded
    if provider in _PROVIDER_MODULES_CACHE:
        return _PROVIDER_MODULES_CACHE[provider]

    # Import and cache the module
    module_name = PROVIDER_MODULES[provider]
    module = __import__(module_name, fromlist=["generate"])  # type: ignore
    generate_func = module.generate

    # Cache for future use
    _PROVIDER_MODULES_CACHE[provider] = generate_func
    return generate_func


def _mongo() -> Tuple[str, str]:
    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DB")
    if not uri or not db:
        raise RuntimeError("MONGODB_URI and MONGODB_DB must be set")
    return uri, db


def _errors_collection_name() -> str:
    return os.getenv("MONGODB_COLLECTION_ERRORS", "errors_cloud")


def _error_rollups_collection_name() -> str:
    return os.getenv("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")


def _jobs_collection_name() -> Optional[str]:
    # Optional; default to 'jobs'. Allow empty to disable.
    return os.getenv("MONGODB_COLLECTION_JOBS", "jobs")


def _log_error_mongo(
    provider: str,
    model: str,
    stage: str,
    message: str,
    tb: Optional[str] = None,
    exc_type: str = "",
    client: Optional[MongoClient] = None,
    db_name: Optional[str] = None,
) -> None:
    try:
        uri, db = _mongo()
        if db_name:
            db = db_name
        coll = _errors_collection_name()

        should_close = False
        if client is None:
            client = MongoClient(uri)
            should_close = True
        try:
            classified = classify_error(message=message, exc_type=exc_type)
            fingerprint = classified.fingerprint(provider=provider, model=model, stage=stage)

            doc = {
                "provider": provider,
                "model_name": model,
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
            client[db][coll].insert_one(doc)

            rollups_coll = client[db][_error_rollups_collection_name()]
            upsert_error_rollup(
                collection=rollups_coll,
                fingerprint=fingerprint,
                provider=provider,
                model_name=model,
                stage=stage,
                error_kind=classified.kind.value,
                normalized_message=classified.normalized_message,
            )
        finally:
            if should_close:
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


def _claim_next_job(client: Optional[MongoClient] = None) -> Optional[Dict]:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return None
    try:
        uri, db_name = _mongo()
    except RuntimeError:
        return None
    
    should_close = False
    if client is None:
        client = MongoClient(uri)
        should_close = True
    
    try:
        jobs = client[db_name][coll_name]
        job = jobs.find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running", "started_at": datetime.now(timezone.utc)}},
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        return job
    finally:
        if should_close:
            client.close()


def _complete_job(job_id, client: Optional[MongoClient] = None) -> None:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return
    
    should_close = False
    if client is None:
        uri, _ = _mongo()
        client = MongoClient(uri)
        should_close = True
        
    _, db_name = _mongo()
    try:
        client[db_name][coll_name].update_one({"_id": job_id}, {"$set": {"status": "done", "finished_at": datetime.now(timezone.utc)}})
    finally:
        if should_close:
            client.close()


def _fail_job(job_id, message: str, client: Optional[MongoClient] = None) -> None:
    coll_name = _jobs_collection_name()
    if not coll_name:
        return

    should_close = False
    if client is None:
        uri, _ = _mongo()
        client = MongoClient(uri)
        should_close = True

    _, db_name = _mongo()
    try:
        client[db_name][coll_name].update_one(
            {"_id": job_id},
            {"$set": {"status": "error", "error": message, "finished_at": datetime.now(timezone.utc)}}
        )
    finally:
        if should_close:
            client.close()


def _validate_metrics(provider: str, metrics: Dict, max_tokens: int) -> Tuple[bool, Optional[str]]:
    if not isinstance(metrics, dict):
        return False, "metrics not a dict"
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for k in required:
        if k not in metrics:
            return False, f"missing metric: {k}"
    if metrics["tokens_per_second"] <= 0:
        return False, "tokens_per_second <= 0"
    out = metrics["output_tokens"]
    if not isinstance(out, int):
        return False, "output_tokens not int"

    # Some providers do not reliably produce exactly max_tokens. Keep strict defaults,
    # but allow known-variable providers to pass if they produce a reasonable amount.
    if provider in ("openai", "openrouter"):
        # These providers can legitimately return fewer than requested tokens (provider-side caps,
        # safety truncation, endpoint-specific behavior). Require non-zero output only.
        if out <= 0:
            return False, f"output_tokens {out} <= 0 for {provider}"
        return True, None

    # Default: require close to requested tokens (±10%).
    if abs(out - max_tokens) > max_tokens * 0.1:
        return False, f"output_tokens {out} not within 10% of requested {max_tokens}"
    return True, None


def _run_single_model(
    provider: str,
    model: str,
    run_always: bool,
    fresh_minutes: int,
    debug: bool,
    client: Optional[MongoClient] = None,
    db_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    # Build configs
    run_ts = get_current_timestamp()
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
        if has_recent_cloud_run(model_name=model, provider=provider, since_utc=since, client=client, db_name=db_name):
            print(f"⏭️ Skipped {provider}:{model} (fresh < {fresh_minutes}m)")
            return {"status": "skipped", "reason": f"fresh < {fresh_minutes}m"}

    # Load provider
    generate = _load_provider_func(provider)

    # Execute and log
    try:
        metrics = generate(model_config, run_config)
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)}"
        _log_error_mongo(
            provider,
            model,
            stage="generate",
            message=reason,
            tb=traceback.format_exc(limit=5),
            exc_type=type(e).__name__,
            client=client,
            db_name=db_name,
        )
        print(f"❌ Error {provider}:{model} - {reason}")
        return {"status": "error", "reason": reason}

    if not metrics:
        msg = "empty metrics"
        _log_error_mongo(provider, model, stage="validate", message=msg, client=client, db_name=db_name)
        print(f"❌ Error {provider}:{model} - {msg}")
        return {"status": "error", "reason": msg}

    ok, why = _validate_metrics(provider, metrics, MAX_TOKENS)
    if not ok:
        _log_error_mongo(provider, model, stage="validate", message=str(why), client=client, db_name=db_name)
        print(f"❌ Error {provider}:{model} - {why}")
        return {"status": "error", "reason": why}

    # Write success only to Mongo metrics collection
    try:
        _log_success_mongo(model_config, metrics)
    except Exception as e:
        reason = f"log failure: {type(e).__name__}: {e}"
        _log_error_mongo(
            provider,
            model,
            stage="log",
            message=reason,
            exc_type=type(e).__name__,
            client=client,
            db_name=db_name,
        )
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
        os.getenv("BENCHMARK_PROVIDERS", "all"),
        "--providers",
        help="Comma-separated list of providers, or 'all' for all available",
    ),
    limit: int = typer.Option(100, "--limit", help="Limit per-provider models"),
    run_always: bool = typer.Option(False, "--run-always", is_flag=True, help="Ignore freshness window for periodic runs"),
    fresh_minutes: int = typer.Option(
        int(os.getenv("FRESH_MINUTES", "30")), 
        "--fresh-minutes", 
        help="Skip runs newer than this window (minutes)"
    ),
    daemon: bool = typer.Option(False, "--daemon", is_flag=True, help="Run as a daemon, checking for jobs and periodic runs"),
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

    # Pick providers
    provider_models: Dict[str, List[str]] = load_provider_models()
    if not providers or providers.strip().lower() == "all":
        selected = list(provider_models.keys())
    else:
        selected = [p.strip() for p in providers.split(",")]

    print(f"Running providers: {selected}")

    if daemon:
        print(f"🚀 Starting in DAEMON mode (cadence={fresh_minutes}m)")
        
        # Calculate the next wall-clock boundary to align the schedule
        # e.g. if fresh_minutes is 30, we want to run at :00 and :30
        now = time.time()
        cadence_seconds = fresh_minutes * 60
        last_periodic_pass = now - (now % cadence_seconds)
        
        while True:
            now = time.time()
            
            # Create one MongoDB client for this check cycle
            client: Optional[MongoClient] = None
            db_name: Optional[str] = None
            try:
                uri, db_name = _mongo()
                client = MongoClient(uri)

                # 0) Recovery: Reset stuck "running" jobs (older than 1 hour)
                coll_name = _jobs_collection_name()
                if coll_name:
                    stale_time = datetime.now(timezone.utc) - timedelta(hours=1)
                    client[db_name][coll_name].update_many(
                        {"status": "running", "started_at": {"$lt": stale_time}},
                        {"$set": {"status": "pending", "started_at": None}}
                    )

                # 1) Always drain ad-hoc jobs first
                while True:
                    job = _claim_next_job(client)
                    if not job:
                        break
                    job_provider = job.get("provider")
                    job_model = job.get("model") or job.get("model_id") or job.get("model_name")
                    ignore_freshness = bool(job.get("ignore_freshness", False))
                    print(f"📥 Found ad-hoc job: {job_provider}:{job_model}")
                    if not job_provider or not job_model:
                        _fail_job(job.get("_id"), "invalid job doc", client)
                        continue
                    
                    # Ad-hoc jobs use a strict freshness check to avoid double-running if a periodic pass just happened
                    res = _run_single_model(job_provider, job_model, run_always or ignore_freshness, fresh_minutes, debug, client, db_name)
                    if res.get("status") == "success":
                        _complete_job(job.get("_id"), client)
                    else:
                        _fail_job(job.get("_id"), res.get("reason") or "unknown error", client)

                # 2) Check if it is time for periodic pass
                if now >= last_periodic_pass + cadence_seconds:
                    # Update to the CURRENT boundary
                    last_periodic_pass = now - (now % cadence_seconds)
                    print(f"⏰ Starting periodic pass (aligned to {datetime.fromtimestamp(last_periodic_pass).strftime('%H:%M:%S')})")
                    
                    # Refresh provider models list
                    provider_models = load_provider_models()
                    for provider in selected:
                        models = provider_models.get(provider, [])[:limit]
                        for model in models:
                            # Force run every model, ignore freshness entirely
                            _run_single_model(provider, model, True, 0, debug, client, db_name)
                    
                    next_run = last_periodic_pass + cadence_seconds
                    print(f"✅ Periodic pass complete. Next run at {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')}")
                
            except Exception as e:
                print(f"⚠️ Daemon loop error: {e}")
                traceback.print_exc()
            finally:
                if client:
                    client.close()

            time.sleep(10)
    else:
        # Standard one-shot execution
        client = None
        db_name = None
        try:
            uri, db_name = _mongo()
            client = MongoClient(uri)

            # 1) Drain ad-hoc jobs
            while True:
                job = _claim_next_job(client)
                if not job:
                    break
                job_provider = job.get("provider")
                job_model = job.get("model") or job.get("model_id") or job.get("model_name")
                ignore_freshness = bool(job.get("ignore_freshness", False))
                if not job_provider or not job_model:
                    _fail_job(job.get("_id"), "invalid job doc", client)
                    continue
                res = _run_single_model(job_provider, job_model, run_always or ignore_freshness, fresh_minutes, debug, client, db_name)
                if res.get("status") == "success":
                    _complete_job(job.get("_id"), client)
                else:
                    _fail_job(job.get("_id"), res.get("reason") or "unknown error", client)

            # 2) Periodic pass
            for provider in selected:
                models = provider_models.get(provider, [])[:limit]
                for model in models:
                    _run_single_model(provider, model, run_always, fresh_minutes, debug, client, db_name)
        finally:
            if client:
                client.close()


if __name__ == "__main__":
    app()
