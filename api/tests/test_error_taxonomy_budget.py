"""Budget exhaustion is a measurement outcome, not a broken model.

Both used to land in `unknown`, where 279 dead letters mixed models the provider
had deleted with models that answered fine and could not be measured in 64
tokens. One of those needs a catalogue fix and the other needs a decision about
the benchmark profile; a single bucket for both hid the second entirely.
"""

from llm_bench.ops.error_taxonomy import ErrorKind
from llm_bench.ops.error_taxonomy import classify_error
from llm_bench.scheduler.policies import should_retry

BUDGET_MESSAGE = "visible output empty after token budget was exhausted; retry with a larger output budget"


def test_budget_exhaustion_is_its_own_kind():
    assert classify_error(message=BUDGET_MESSAGE).kind is ErrorKind.BUDGET_EXHAUSTED


def test_budget_exhaustion_is_not_retried():
    # Nothing about the next attempt differs — same model, same 64-token
    # budget — so a retry buys an identical result at provider prices.
    assert should_retry("budget_exhausted", attempt=1, max_attempts=3) is False


def test_a_dedicated_endpoint_refusal_is_a_hard_model():
    # Together answers 400, not 404, for a model the account cannot call. Left
    # as `unknown` it was retried on every pass forever.
    message = (
        "BadRequestError: Error code: 400 - {'error': {'message': 'The dedicated endpoint for "
        "minimaxai/minimax-m1-40k is not available'}}"
    )
    assert classify_error(message=message).kind is ErrorKind.HARD_MODEL


def test_a_retired_bedrock_version_is_a_hard_model():
    """Bedrock answers ResourceNotFoundException with no status to key on.

    All three Bedrock models that had gone quiet turned out to be saying this,
    and nobody could see it because the runner had no way to report a failure.
    """
    message = (
        "ResourceNotFoundException: An error occurred (ResourceNotFoundException) when calling the "
        "ConverseStream operation: This model version has reached the end of its life."
    )
    assert classify_error(message=message).kind is ErrorKind.HARD_MODEL


def test_a_deleted_model_is_still_unknown_here():
    """OpenAI's 400 for a deleted model has no status signal to key on.

    Left to the LLM classifier by design. Pinned so that if someone teaches the
    deterministic path to recognise it, they do it on purpose.
    """
    message = "BadRequestError: Error code: 400 - {'error': {'message': \"The requested model 'x' does not exist.\"}}"
    assert classify_error(message=message).kind is ErrorKind.UNKNOWN


def test_real_failures_keep_their_classification():
    assert classify_error(message="Error code: 404 - not found").kind is ErrorKind.HARD_MODEL
    assert classify_error(message="Error code: 429 - slow down").kind is ErrorKind.RATE_LIMIT
    assert classify_error(message="Error code: 401 - bad key").kind is ErrorKind.AUTH


def test_openrouter_monthly_key_limit_is_billing_not_auth():
    message = "PermissionDeniedError: Error code: 403 - {'error': {'message': 'Key limit exceeded (monthly limit).'}}"
    assert classify_error(message=message).kind is ErrorKind.BILLING
