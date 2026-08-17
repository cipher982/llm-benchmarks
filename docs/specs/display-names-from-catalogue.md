# Display names from the OpenRouter catalogue

**Status:** reviewed (codex luna, xhigh, 2026-08-17). Decision recorded. Not
implemented.

All counts below are a live snapshot of 2026-08-17 and are dated on purpose —
the first draft quoted numbers that did not survive checking.

## Problem

The Delivered TPS leaderboard has 236 rows. 195 of them render a string
containing a slash — what a reader sees as a raw id (`mistralai/ministral-14b-2512`,
`aion-labs/aion-2.0-20260223`). Of those, 115 are *exactly* the model id and the
rest are dated machine slugs. Both are failures of presentation; the slash count
is the one that matches what a visitor perceives.

The site went from ~60 hand-named models across nine providers to 388 across
four, and the naming layer did not come with it.

## Root cause

Not missing data. One line, at `ops/openrouter_discovery.py:81-84`:

```python
def _display_name(row, model_id):
    """Prefer OpenRouter's canonical display slug without changing identity."""
    return str(row.get("canonical_slug") or row.get("name") or model_id)
```

`canonical_slug` is the dated machine slug (`z-ai/glm-4.7-20251222`). `name` is
the human label (`Z.ai: GLM 4.7`). This prefers the slug and calls it a "display
slug" in its own docstring. `ops/admission.py:181,204` then copies that into
`models.display_name`, which is what the dashboard renders.

So a backfill alone would be undone: the continuous path must be fixed first, or
it keeps writing the wrong label.

## What the catalogue actually provides

`openrouter_catalog`, refreshed by discovery. Coverage is **not** universal, and
the collection is append-only — rows are upserted and never removed or marked
absent (`openrouter_discovery.py:111-116,164-181`), so the total is historical,
not current:

| field | rows | note |
|---|---|---|
| total documents | 619 | historical; only 421 seen since the latest run cutoff |
| `name` | 619 | but only 577 contain `": "` and are splittable |
| `org` | 614 | |
| `canonical_slug` | 421 | |
| `base_model_id` | 343 | |

`name` is `"{vendor}: {model}"`, the same pair the OpenRouter rankings page
renders. It keeps a date only where the vendor versions that way
(`DeepSeek V4 Flash 0731`), which is the desired behaviour rather than a defect.

No LLM call and no scraping. The field is already stored and simply not read.

## Decision: group on identity, not on the display string

The publication pipeline groups by `(providerCanonical, display_name,
transportProvider)` as an exact string match (`modelMappingDB.ts:214-217`), and
the time-series layer groups by `model_name` alone (`dataProcessing.ts:204-212`).
Changing display names therefore moves chart identity, which is the wrong
coupling: `claude-haiku-4.5` and `claude-haiku-4-5` already split one
three-provider model into two lines once.

Three options were considered — normalise strings for grouping, rename direct
lanes to agree, or stop grouping on the string. **Take the third.**

`bench_model_identity` already exists and already defines `canonical_key` as the
grouping identity (`ops/identity.py:56-93`), maintained continuously by the
reconciler (`ops/reconciler.py:161-200`). The abstraction is built; the
dashboard just cannot see it. Normalising strings is a heuristic that will
eventually false-merge, and renaming direct lanes leaves mutable presentation
text acting as identity.

Two levels, deliberately separate:

- **Lane aggregation:** `(providerCanonical, modelCanonical, transportProvider)`.
- **Cross-provider chart identity:** `canonical_key`.

Raw samples must not be grouped by `canonical_key` directly — the identity layer
can place two endpoints from one provider in the same group, which the
reconciler already guards against (`reconciler.py:278-305`).

Exposure is smaller than the draft claimed: identity currently finds 10 Bedrock
and 2 Vertex enabled endpoints in groups that also contain an OpenRouter
endpoint. Identity coverage is incomplete, so that is a lower bound.

## Label source order

`unify_display_names` (`reconciler.py:243-256`) already gives every endpoint in a
group one name, but picks *the name most of the group already uses* — so a good
catalogue label can lose to a bad hand-written one. It needs an explicit
precedence instead:

1. Current OpenRouter catalogue label for a group with an OpenRouter member.
2. Provider catalogue label where the provider supplies one.
3. The identity resolver's group label — which today is slugified away
   (`identity.py:145-163`) and must be persisted as readable text.
4. An explicitly marked canonical-id fallback.

Direct-only models may still render a normalised id. That is better than
reintroducing a hand-maintained table, which is the thing being retired.
Vertex discovery writes `name = model_id` (`vertex_discovery.py:92-149`) and the
Bedrock catalogue is an id list (`cloud/models.json:24-60`), so no feed exists
for those.

## Failure modes to handle

- **Renames.** Must move the label, never `canonical_key`.
- **Stale catalogue rows.** Absence from a refresh must not overwrite a known
  label; use only rows from the latest complete discovery run.
- **Collisions.** Stripping the vendor prefix is not collision-safe: nine
  duplicate parsed labels across 18 rows, including cross-organisation ones like
  `Reka Edge`. Same-provider collisions already exist (`gpt-4` twice).
- **Variant families.** 64 `:free` and 61 `:batch` rows. Discovery deliberately
  collapses suffixed variants to one base row
  (`openrouter_discovery.py:38-55`); the spec must state whether a variant is a
  benchmark identity or a service variant. The `(free)` suffix in the display
  name does not make this safe.
- **Non-text and deprecated.** 198 historical rows carry empty or non-text
  modality metadata and the catalogue has no deprecation field; publication must
  key off `models.enabled/deprecated` and discovery-run freshness.
- **`vendor` is not wired anywhere.** `ModelSchema` has no such field
  (`modelMappingDB.ts:9-22`), the projection omits it, and `CloudBenchmark` does
  not carry it (`CloudData.ts:1-28`). Either wire it through the contract or
  drop it from this spec.

## Implementation order

1. Report-only audit: for every enabled model, classify exact / base / stale /
   ambiguous / missing catalogue match, parsed-label collisions, and identity
   coverage. Derive the numbers; do not hardcode the ones above.
2. Fix `_display_name` to prefer the parsed `name`, with slug and id as
   fallbacks. Without this the continuous path keeps overwriting.
3. One authoritative catalogue-label helper: current text-capable rows only,
   preserving `name`, `org`, `base_model_id`, `canonical_slug`, `last_seen_at`,
   rejecting stale rows.
4. Reversible backfill of labels through a mutation batch. A single broken row
   (`vertex undefined`) fails the audit, not the batch.
5. Backfill `bench_model_identity` for all enabled endpoints via the existing
   bounded resolver, and persist a readable group label alongside
   `canonical_key`.
6. Add `identityKey` to the dashboard metadata contract. Canonical ids and slugs
   unchanged — slugs already derive from canonical fields
   (`deliveredTpsProcessing.ts:107-115`, `modelMappingMerge.ts:56-65`), so no URL
   moves.
7. Move lane aggregation off display name; attach `identityKey` per lane.
8. Move time-series grouping to `identityKey`, with an endpoint-specific
   fallback for unresolved or same-provider collisions.
9. Keep `transportProvider` on every lane — direct and routed samples stay
   separate aggregates (`transportGrouping.test.js:21-35`).
10. Run the naming pass immediately after discovery/admission rather than as an
    independent six-hour loop (`scheduler/cli.py:538-589`), and invalidate the
    dashboard's five-minute cache after mutation (`cache.ts:15-41`).
11. Retire `unify_display_names` as the grouping mechanism once identity-key
    publication is live.

## Tests that would actually pin this

- Label parsing: normal, no separator, missing name, vendor containing a colon.
- Freshness filtering, and a truncated or failed discovery run.
- `:free`, `:batch`, unsuffixed and dated variants.
- Two providers, different labels, same `canonical_key` → one chart model.
- Two endpoints from one provider with the same identity → do not merge.
- Direct and routed samples stay separate.
- An OpenRouter rename moves the label but not chart identity or slugs.
- Unresolved endpoints stay separate and record an explicit fallback source.
- Backfill is reversible; the continuous update is idempotent.
- Live invariant: every enabled row has a sourced label or a recorded fallback.

## Open question

Whether `display_name` stays endpoint-specific while charts use a separate group
label. Recommended: yes — it lets OpenRouter rename its own presentation without
rewriting every provider row, while the chart stays stable on `canonical_key`.
