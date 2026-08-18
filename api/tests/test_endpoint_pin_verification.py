"""A pin is confirmed by the name OpenRouter gave the endpoint at discovery.

The response echoes a provider display name and never the endpoint tag, so the
display name is the finest identity a completed generation can confirm. The
reviewed slug allowlist maps display names onto our own direct-lane slugs --
a migration concern -- and knows 9 providers. Against a 693-endpoint catalogue
spanning 65 providers it rejected 550 endpoints as unverifiable, which the
runner then served as unpinned fallbacks.
"""

from llm_bench.cloud.providers.openrouter import _observed_provider


def _metadata(display):
    return {"provider": display}


class TestDisplayNameConfirmsAPin:
    def test_unlisted_provider_is_verifiable_via_discovery(self):
        provider, slug = _observed_provider(
            _metadata("Novita"),
            expected_slug="novita",
            expected_display="Novita",
            verify_catalog=True,
        )
        assert (provider, slug) == ("Novita", "novita")

    def test_naming_drift_between_display_and_canonical_still_matches(self):
        # "AionLabs" is the display name; "aion-labs" is the canonical slug.
        # Normalized comparison is what keeps this from being the org-prefix
        # blindness that hid 37 real models in the consolidation pass.
        provider, slug = _observed_provider(
            _metadata("AionLabs"),
            expected_slug="aion-labs",
            expected_display="Aion Labs",
            verify_catalog=True,
        )
        assert slug == "aion-labs"

    def test_a_different_provider_is_not_confirmed(self):
        # OpenRouter served someone other than who we pinned: no slug, so
        # provider_metadata_verified stays false and the row is rejected.
        _, slug = _observed_provider(
            _metadata("Parasail"),
            expected_slug="novita",
            expected_display="Novita",
            verify_catalog=True,
        )
        assert slug is None

    def test_allowlisted_providers_still_resolve_without_a_display_name(self):
        provider, slug = _observed_provider(_metadata("Groq"), expected_slug="groq", verify_catalog=True)
        assert (provider, slug) == ("Groq", "groq")
