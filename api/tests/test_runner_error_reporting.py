"""The Bedrock runner must report failures, not just log them.

Three Bedrock models failed on roughly 657 consecutive cycles across fourteen
days and `errors_cloud` held nothing. Every failure path in the runner wrote to
a log file on an EC2 instance and returned False. The models did not look
broken; they had no recent data, and only a coverage invariant over the desired
set ever noticed.
"""

from unittest.mock import patch

import bench_simple_runner as runner
from llm_bench.config import CloudConfig


def _raises(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _config():
    return CloudConfig(
        provider="bedrock",
        model_name="us.meta.llama3-2-90b-instruct-v1:0",
        run_ts="2026-08-05 01:00:00",
        temperature=0.1,
    )


def test_a_provider_exception_is_reported():
    reported = []
    with (
        patch.object(
            runner, "_load_provider_func", return_value=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
        patch.object(runner, "log_http_error", side_effect=lambda cfg, **kw: reported.append(kw)),
    ):
        assert runner.run_single_benchmark("bedrock", "us.meta.llama3-2-90b-instruct-v1:0") is False

    assert len(reported) == 1
    assert reported[0]["stage"] == "generate"
    assert "RuntimeError" in reported[0]["message"]


def test_empty_metrics_are_reported():
    reported = []
    with (
        patch.object(runner, "_load_provider_func", return_value=lambda *a, **k: {}),
        patch.object(runner, "log_http_error", side_effect=lambda cfg, **kw: reported.append(kw)),
    ):
        assert runner.run_single_benchmark("bedrock", "amazon.nova-pro-v1:0") is False

    assert reported[0]["stage"] == "generate"
    assert "Empty metrics" in reported[0]["message"]


def test_a_validation_failure_is_reported():
    bad = {"output_tokens": 0, "generate_time": 1.0, "tokens_per_second": 0}
    reported = []
    with (
        patch.object(runner, "_load_provider_func", return_value=lambda *a, **k: bad),
        patch.object(runner, "log_http_error", side_effect=lambda cfg, **kw: reported.append(kw)),
    ):
        assert runner.run_single_benchmark("bedrock", "amazon.nova-pro-v1:0") is False

    assert reported[0]["stage"] == "validate"


def test_a_rejected_ingest_is_reported():
    good = {"output_tokens": 64, "generate_time": 1.0, "tokens_per_second": 64.0}
    reported = []
    with (
        patch.object(runner, "_load_provider_func", return_value=lambda *a, **k: good),
        patch.object(runner, "log_http", return_value=False),
        patch.object(runner, "log_http_error", side_effect=lambda cfg, **kw: reported.append(kw)),
    ):
        assert runner.run_single_benchmark("bedrock", "amazon.nova-pro-v1:0") is False

    assert reported[0]["stage"] == "ingest"


def test_a_success_reports_nothing():
    good = {"output_tokens": 64, "generate_time": 1.0, "tokens_per_second": 64.0}
    reported = []
    with (
        patch.object(runner, "_load_provider_func", return_value=lambda *a, **k: good),
        patch.object(runner, "log_http", return_value=True),
        patch.object(runner, "log_http_error", side_effect=lambda cfg, **kw: reported.append(kw)),
    ):
        assert runner.run_single_benchmark("bedrock", "amazon.nova-pro-v1:0") is True

    assert reported == []


def test_the_error_url_is_derived_from_the_ingest_url():
    """Not separately configured, so a runner cannot be set up to post metrics
    while silently dropping every failure."""
    from llm_bench import http_output

    with patch.dict("os.environ", {"INGEST_API_URL": "https://bench-ingest.drose.io/ingest"}, clear=False):
        assert http_output._error_url() == "https://bench-ingest.drose.io/ingest/error"


def test_the_reported_kind_comes_from_the_shared_taxonomy():
    from llm_bench import http_output

    sent = {}
    with (
        patch.dict(
            "os.environ",
            {"INGEST_API_URL": "https://example.invalid/ingest", "INGEST_API_KEY": "k"},
            clear=False,
        ),
        patch.object(http_output.httpx, "post", side_effect=lambda url, **kw: sent.update(kw["json"]) or _ok()),
    ):
        http_output.log_http_error(_config(), message="Error code: 404 - not found", stage="generate")

    assert sent["error_kind"] == "hard_model"


class _ok:
    status_code = 200

    def raise_for_status(self):
        return None
