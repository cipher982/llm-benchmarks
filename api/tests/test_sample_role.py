"""The boundary between what is measured and what is published.

Before this existed, a successful probe was indistinguishable from a published
measurement: same collection, same freshness credit. Probe-before-promote needs
to make real calls against candidate models, so without this the admission
process would contaminate the series it exists to protect.
"""

from llm_bench.scheduler import runner
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import probe_metrics_collection_name


class TestSampleRouting:
    def test_published_samples_go_to_the_public_collection(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        captured = {}
        monkeypatch.setattr(runner, "log_mongo", lambda **kw: captured.update(kw))
        runner.log_success_mongo(object(), {}, sample_role=runner.SAMPLE_ROLE_PUBLISHED)
        assert captured["collection_name"] == metrics_collection_name()

    def test_probe_and_shadow_samples_do_not(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        for role in (runner.SAMPLE_ROLE_PROBE, runner.SAMPLE_ROLE_SHADOW):
            captured = {}
            monkeypatch.setattr(runner, "log_mongo", lambda **kw: captured.update(kw))
            runner.log_success_mongo(object(), {}, sample_role=role)
            assert captured["collection_name"] == probe_metrics_collection_name(), role

    def test_routing_is_by_collection_not_by_a_flag_readers_must_remember(self):
        """The two collections must actually differ.

        Filtering on a field would put the burden on every one of the
        dashboard's read paths; missing one publishes probe data silently.
        """
        assert probe_metrics_collection_name() != metrics_collection_name()


class TestProvenance:
    def test_a_default_job_is_published_with_full_provenance(self, monkeypatch):
        written = {}

        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        monkeypatch.setattr(
            runner,
            "log_success_mongo",
            lambda config, metrics, *, sample_role: written.update(metrics, _role=sample_role),
        )

        result = runner.run_benchmark_job({"_id": "job-1", "provider": "groq", "model_id": "m", "attempt": 3})

        assert result.status == "success"
        assert result.sample_role == runner.SAMPLE_ROLE_PUBLISHED
        assert written["sample_role"] == runner.SAMPLE_ROLE_PUBLISHED
        assert written["protocol_version"] == runner.PROTOCOL_VERSION
        assert written["benchmark_profile_id"] == runner.DEFAULT_PROFILE_ID
        # The spike found 975 of 1,182 Together rows came from multi-attempt
        # retries with nothing on the row to say so.
        assert written["attempt_group"] == "job-1"
        assert written["attempt"] == 3

    def test_a_probe_job_carries_its_role_out_to_the_caller(self, monkeypatch):
        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        monkeypatch.setattr(runner, "log_success_mongo", lambda *a, **k: None)

        result = runner.run_benchmark_job(
            {"_id": "job-2", "provider": "groq", "model_id": "m", "sample_role": runner.SAMPLE_ROLE_PROBE}
        )

        # The worker reads this to decide whether the run counts as freshness.
        assert result.sample_role == runner.SAMPLE_ROLE_PROBE
        assert result.sample_role in runner.NON_PUBLISHING_ROLES


METRICS = {"output_tokens": 64, "generate_time": 1.0, "tokens_per_second": 64.0}


class TestProvenanceSurvivesTheWriteLayer:
    """The mocked tests above cannot see the allowlist that drops fields.

    log_mongo copies only keys named in OPTIONAL_METRIC_FIELDS. The first
    deploy of this contract set all five provenance fields and wrote 92 rows
    carrying none of them, because a test that mocks the writer proves only
    that the caller passed something.
    """

    def test_every_provenance_field_is_on_the_allowlist(self):
        from llm_bench.logging import OPTIONAL_METRIC_FIELDS

        required = {
            "sample_role",
            "benchmark_profile_id",
            "protocol_version",
            "attempt_group",
            "attempt",
            "profile_hash",
            "effective_request_hash",
            "route_revocation_generation",
        }
        assert required <= set(OPTIONAL_METRIC_FIELDS), required - set(OPTIONAL_METRIC_FIELDS)

    def test_the_fields_the_runner_sets_are_exactly_the_fields_that_persist(self, monkeypatch):
        from llm_bench.logging import _optional_metric_fields

        monkeypatch.setattr(runner, "load_provider_func", lambda p: lambda cfg, rc: dict(METRICS))
        written = {}
        monkeypatch.setattr(
            runner,
            "log_success_mongo",
            lambda config, metrics, *, sample_role: written.update(metrics),
        )
        runner.run_benchmark_job({"_id": "job-1", "provider": "groq", "model_id": "m", "attempt": 2})

        persisted = _optional_metric_fields(written)
        for field in (
            "sample_role",
            "benchmark_profile_id",
            "protocol_version",
            "attempt_group",
            "attempt",
        ):
            assert field in persisted, f"{field} is set by the runner but dropped before Mongo"
        for field in (
            "profile_hash",
            "effective_request_hash",
            "direct_effective_request_hash",
            "routed_effective_request_hash",
            "route_revocation_generation",
        ):
            if field in written:
                assert field in persisted, f"{field} is set by the runner but dropped before Mongo"
