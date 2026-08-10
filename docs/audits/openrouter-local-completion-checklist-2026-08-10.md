# OpenRouter local completion checklist

Date: 2026-08-10
Scope: finish the OpenRouter consolidation locally, then stop before production deployment.

## Objective

Close the frozen 241-row enabled-model inventory with evidence. Route a source
row through OpenRouter only when its identity, provider pin, protocol, pricing,
availability, and paired canary all pass. Keep the existing direct provider
lane for every other row. A route candidate is never an active route.

## Worklist

- [x] Freeze and verify the 241-row source input and the current OpenRouter
  evidence bundle.
- [x] Review every candidate by source provider. Preserve all source-row to
  OpenRouter-ID mappings, including duplicates that share one target ID.
- [x] Fill the missing direct and routed pricing evidence for each eligible
  candidate. Do not treat catalog pricing alone as a canary result.
- [x] Run bounded paired direct/OpenRouter canaries for every eligible
  candidate. Record request hashes, provider observations, output validity,
  latency, throughput, cost, errors, and confidence intervals.
- [x] Retry transient probe or canary failures within the budget. Classify a
  persistent failure as direct with a concrete reason. Do not leave work in a
  candidate-only state.
- [x] Handle policy and transport exceptions explicitly: Bedrock remains
  direct under the current policy, Vertex and image/modality cases require
  their own compatibility evidence, and hard endpoint mismatches remain
  direct.
- [x] Generate one terminal decision for each of the 241 source rows. The
  report must have no dropped rows, duplicate active routes, or unresolved
  `route-candidate` records.
- [x] Materialize the locally active route map from passing canaries only.
  Keep direct credentials, adapters, and fallback behavior for every source
  row.
- [x] Enable the route map in the local runtime or test harness only. Verify
  that approved rows use OpenRouter and all other rows use direct transport,
  with automatic fallback on route failure.
- [x] Run the API regression suite, focused routing tests, smoke requests,
  artifact-manifest verification, packaging checks, and a clean local startup.
- [x] Obtain independent final review receipts from Hatch Sol and Cursor Grok.
- [x] Record the final counts, hashes, test commands, reviewer receipts, and
  the exact production gate. Stop before any production Mongo write,
  deployment, or feature-flag change.

## Success criteria

The task is complete when all of these are true:

1. The final reconciliation contains exactly 241 unique source rows, and each
   row has exactly one terminal state and a concrete reason.
2. No row remains only because the canary loop stopped. Unfinished evidence is
   either completed or explicitly classified as direct with a recorded cause,
   retry result, and recheck time.
3. Every `route-approved` row has an exact or reviewed OpenRouter identity,
   primary identity evidence, endpoint/provider evidence, pricing evidence,
   successful pinned probes, and a passing paired canary.
4. The local route map contains only approved routes, and its source and
   evidence hashes match the final reconciliation.
5. A local request for an approved row uses OpenRouter; a request for any
   direct row uses its original provider; and a forced OpenRouter error returns
   the row to direct transport without losing the source identity.
6. The API tests, focused routing tests, smoke tests, packaging checks, and
   artifact verification pass from the checked-out repository.
7. Direct and routed metrics remain separate, and no production state has been
   changed. Production activation remains a separate explicit step after this
   checkpoint.
8. Hatch Sol and Cursor Grok independently review the implementation and
   return ready/approved receipts, with any findings resolved or recorded.

## Explicit stop gate

This run may write local files and local test/runtime state. It may not write
production MongoDB, enable a production routing flag, deploy clifford, or
change the Bedrock runner. The handoff must include the final reconciliation
counts and the exact command a human would run to cross that production gate.

## Review receipts

- Hatch Sol: READY, run `hatch_20260810T225340.288385000Z_3914ce2ba11b56f0`.
- Cursor Grok: READY, run `hatch_20260810T225654.179453000Z_55b59b431b9c7963`.
