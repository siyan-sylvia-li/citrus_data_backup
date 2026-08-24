"""The two model facts that more than one file needs.

Everything else builds its own `dspy.LM` inline, where the provider is visible at the
call site.

1. Which Claude model the annotation layers use, and on which provider.
2. The three-judge screening panel, shared by prompt_filter.JudgeSuite and
   intervention_filter.EffortSuite, which are required to behave and fail identically
   (see intervention_filter's docstring) and which used to hold two copies of it.

Claude is on the ANTHROPIC API (ANTHROPIC_API_KEY). Bedrock is intended to replace that
later; the notes below record what was already established, so the switch does not have
to be re-derived.

    Bedrock findings, from a working session against this account
    ------------------------------------------------------------
    * The credential in .env (AWS_BEARER_TOKEN_BEDROCK) is a MANTLE gateway token, not a
      sigv4 pair. It is scoped to us-east-2 -- every other region 401s with "Credential
      should be scoped to a valid region".
    * Claude on Mantle is reached through the Anthropic Messages API, not the
      OpenAI-compatible /v1/chat/completions path:

          AnthropicBedrockMantle(
              aws_region="us-east-2",
              api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
              default_headers={"anthropic-workspace-id": "proj_..."},
          )

      The workspace header is honoured on that path (it changes the resource the IAM
      check names) and REJECTED on the OpenAI-compatible one.
    * dspy cannot currently reach it. litellm 1.81.0 lists `bedrock_mantle` in its cost
      map but implements no such provider, so `dspy.LM("bedrock_mantle/...")` raises.
      Reaching Mantle from dspy needs a custom LM wrapping the client above. litellm's
      `bedrock/` provider is a DIFFERENT service (bedrock-runtime) and will not accept a
      Mantle token.
    * Still blocked account-side: an explicit deny on bedrock-mantle:CreateInference from
      the IAM policy RequiredTagsPolicy. An explicit deny cannot be overridden by any
      client configuration.
"""
from __future__ import annotations

import dspy

# One place to name the model, because two annotation layers use it.
ANTHROPIC_MODEL = "anthropic/claude-sonnet-5"


def anthropic_model(model: str = "claude-sonnet-5") -> str:
    """The litellm model string for a Claude model.

    Separate from `anthropic_lm` because intervention_pretest builds its readers from a
    list of bare model strings.
    """
    return f"anthropic/{model}"


def anthropic_lm(model: str = "claude-sonnet-5", **kw) -> dspy.LM:
    """A Claude model on the Anthropic API.

    Never pass a temperature: current Claude models 400 on any sampling parameter. dspy
    forwards only kwargs that are not None, so omitting it is enough.
    """
    return dspy.LM(anthropic_model(model), **kw)


# name -> (lm, dspy module class). No Anthropic model sits in this panel, so none of it
# goes through Bedrock; the name is RECORDED per participant as `judges[].model`.
#
# The Qwen3-235B judge went off serverless; a sweep of the Together serverless catalog
# found Nemotron the best replacement third judge -- most decorrelated from Llama+GPT, so
# it wins on panel agreement (screener ICC 0.685 -> 0.743, r 0.785 -> 0.841, exact
# 50% -> 65%). gpt-oss-120b / MiniMax-M3 tie it on r but lose on ICC/exact; small Qwen and
# big reasoning models were too weak or too slow.
#
# timeout caps a slow/straggling model; on timeout that grader errors, is dropped, and the
# panel returns from the others (fail-open).
def panel_specs() -> dict[str, tuple]:
    return {
        "llama": (dspy.LM("together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                          temperature=0.0, max_tokens=2048, timeout=10), dspy.Predict),
        "gpt": (dspy.LM("openai/gpt-5.4-mini", max_tokens=2048, timeout=10),
                dspy.ChainOfThought),
        "nemotron": (dspy.LM("together_ai/nvidia/nemotron-3-ultra-550b-a55b",
                             temperature=0.0, max_tokens=2048, timeout=15), dspy.Predict),
    }
