from llm_bench.scheduler import runner


def active_snapshot(**overrides):
    value = {
        "source_provider": "deepinfra",
        "source_model_id": "Qwen/Qwen3-32B",
        "route_decision_version": "or-route-v1",
        "state": "active",
        "transport_provider": "openrouter",
        "route_policy": "pinned-provider",
        "route_model_id": "qwen/qwen3-32b",
        "route_provider_slug": "deepinfra",
        "observed_provider": "DeepInfra",
        "observed_provider_slug": "deepinfra",
        "provider_metadata_verified": True,
        "route_snapshot_at": "2026-08-09T00:00:00+00:00",
        "route_probe_id": "probe-1",
    }
    value.update(overrides)
    return value


def metrics(**overrides):
    value = {"output_tokens": 64, "generate_time": 1.0, "tokens_per_second": 64.0}
    value.update(overrides)
    return value


def job(**overrides):
    value = {
        "_id": "job-1",
        "provider": "deepinfra",
        "model_id": "Qwen/Qwen3-32B",
        "attempt": 1,
        "route_snapshot": active_snapshot(),
    }
    value.update(overrides)
    return value


def test_active_route_is_opt_in_and_preserves_source_identity(monkeypatch):
    calls = []
    configs = []
    written = {}

    def load(provider):
        calls.append(provider)

        def generate(config, run_config):
            configs.append(config)
            return metrics(
                observed_provider="DeepInfra",
                observed_provider_slug="deepinfra",
                provider_metadata_verified=True,
            )

        return generate

    monkeypatch.setenv("OPENROUTER_ROUTING_ENABLED", "1")
    monkeypatch.setattr(runner, "load_provider_func", load)
    monkeypatch.setattr(
        runner,
        "log_success_mongo",
        lambda config, metrics, *, sample_role: written.update(config=config, metrics=metrics),
    )

    result = runner.run_benchmark_job(job())

    assert result.status == "success"
    assert result.transport_provider == "openrouter"
    assert result.route_attempted is True
    assert calls == ["openrouter"]
    assert configs[0].provider == "openrouter"
    assert configs[0].model_name == "qwen/qwen3-32b"
    assert configs[0].source_provider == "deepinfra"
    assert configs[0].source_model_id == "Qwen/Qwen3-32B"
    assert written["metrics"]["source_provider"] == "deepinfra"
    assert written["metrics"]["transport_provider"] == "openrouter"
    assert written["metrics"]["route_model_id"] == "qwen/qwen3-32b"


def test_route_snapshot_stays_direct_when_activation_is_disabled(monkeypatch):
    calls = []
    written = {}

    def load(provider):
        calls.append(provider)
        return lambda config, run_config: metrics()

    monkeypatch.delenv("OPENROUTER_ROUTING_ENABLED", raising=False)
    monkeypatch.setattr(runner, "load_provider_func", load)
    monkeypatch.setattr(
        runner,
        "log_success_mongo",
        lambda config, metrics, *, sample_role: written.update(config=config, metrics=metrics),
    )

    result = runner.run_benchmark_job(job())

    assert result.status == "success"
    assert result.transport_provider == "direct"
    assert result.route_attempted is False
    assert calls == ["deepinfra"]
    assert written["metrics"]["route_reason"] == "route-activation-disabled"
    assert written["metrics"]["transport_provider"] == "direct"


def test_route_failure_gets_a_separate_direct_recovery_attempt(monkeypatch):
    calls = []
    errors = []
    written = {}

    def load(provider):
        calls.append(provider)
        if provider == "openrouter":

            def fail(config, run_config):
                raise RuntimeError("route unavailable")

            return fail
        return lambda config, run_config: metrics()

    monkeypatch.setenv("OPENROUTER_ROUTING_ENABLED", "true")
    monkeypatch.setattr(runner, "load_provider_func", load)
    monkeypatch.setattr(runner, "log_error_mongo", lambda **kwargs: errors.append(kwargs) or "provider_error")
    monkeypatch.setattr(
        runner,
        "log_success_mongo",
        lambda config, metrics, *, sample_role: written.update(config=config, metrics=metrics),
    )

    result = runner.run_benchmark_job(job())

    assert result.status == "success"
    assert result.transport_provider == "direct"
    assert result.route_attempted is True
    assert result.fallback_reason == "RuntimeError: route unavailable"
    assert calls == ["openrouter", "deepinfra"]
    assert errors[0]["stage"] == "route_generate"
    assert written["metrics"]["transport_provider"] == "direct"
    assert written["metrics"]["fallback_reason"] == "RuntimeError: route unavailable"
    assert written["config"].source_provider == "deepinfra"
    assert written["config"].model_name == "Qwen/Qwen3-32B"


def test_route_provider_mismatch_recovers_direct(monkeypatch):
    calls = []
    errors = []
    written = {}

    def load(provider):
        calls.append(provider)
        if provider == "openrouter":
            return lambda config, run_config: metrics(
                observed_provider="Together",
                observed_provider_slug="together",
                provider_metadata_verified=True,
            )
        return lambda config, run_config: metrics()

    monkeypatch.setenv("OPENROUTER_ROUTING_ENABLED", "1")
    monkeypatch.setattr(runner, "load_provider_func", load)
    monkeypatch.setattr(runner, "log_error_mongo", lambda **kwargs: errors.append(kwargs) or "provider_error")
    monkeypatch.setattr(
        runner,
        "log_success_mongo",
        lambda config, metrics, *, sample_role: written.update(config=config, metrics=metrics),
    )

    result = runner.run_benchmark_job(job())

    assert result.status == "success"
    assert calls == ["openrouter", "deepinfra"]
    assert errors[0]["stage"] == "route_validate"
    assert "does not match route" in errors[0]["message"]
    assert written["metrics"]["transport_provider"] == "direct"


def test_direct_failure_does_not_call_another_provider(monkeypatch):
    calls = []
    errors = []

    def load(provider):
        calls.append(provider)

        def fail(config, run_config):
            raise RuntimeError("direct down")

        return fail

    monkeypatch.setattr(runner, "load_provider_func", load)
    monkeypatch.setattr(runner, "log_error_mongo", lambda **kwargs: errors.append(kwargs) or "provider_error")

    result = runner.run_benchmark_job(
        {
            "_id": "job-direct",
            "provider": "deepinfra",
            "model_id": "Qwen/Qwen3-32B",
            "attempt": 1,
        }
    )

    assert result.status == "error"
    assert calls == ["deepinfra"]
    assert errors[0]["stage"] == "generate"
