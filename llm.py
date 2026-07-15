"""
Thin wrapper that returns parsed JSON from either Google Gemini or OpenAI.
Both provider functions expose the same call_json(system, user, max_tokens)
signature, so agents.py and orchestrator.py need no changes regardless of
which provider is active.

Setup:
    OpenAI (default):
        pip install openai
        set OPENAI_API_KEY

    Gemini:
        pip install google-genai
        set GEMINI_API_KEY
        set LLM_PROVIDER=gemini
"""
import json
import os
import re
import time

PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

# Gemini: "gemini-2.5-flash" is no longer available to new API keys (404).
# "gemini-flash-latest" works but its free daily quota (20 req/day at time
# of writing) is easy to exhaust; "gemini-flash-lite-latest" is a separate
# quota bucket with a much higher free-tier allowance.
GEMINI_MODEL = "gemini-flash-lite-latest"

OPENAI_MODEL = "gpt-5.4-mini"

_gemini_client = None
_openai_client = None

# Running token totals for the active model, for cost-comparison tooling
# (e.g. model_comparison.py). Not used by the normal agent/orchestrator path.
_usage_totals = {"input_tokens": 0, "output_tokens": 0}


def reset_usage():
    """Zero the running token counters. Call before timing a comparison run."""
    _usage_totals["input_tokens"] = 0
    _usage_totals["output_tokens"] = 0


def get_usage():
    """Return {"input_tokens": int, "output_tokens": int} accumulated since
    the last reset_usage() call."""
    return dict(_usage_totals)


def _record_usage(input_tokens, output_tokens):
    _usage_totals["input_tokens"] += input_tokens or 0
    _usage_totals["output_tokens"] += output_tokens or 0


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _call_gemini(system, user, max_tokens):
    from google.genai import types

    client = _get_gemini_client()
    last_error = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                    # gemini-flash-latest "thinks" before answering by default,
                    # and thinking tokens are drawn from max_output_tokens --
                    # left enabled, replies get cut off before any JSON is
                    # emitted. Disable it; this is a JSON-extraction task, not
                    # one that benefits from extended reasoning.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            usage = resp.usage_metadata
            if usage is not None:
                _record_usage(usage.prompt_token_count, usage.candidates_token_count)
            text = resp.text
            text = re.sub(r"```json|```", "", text).strip()
            return json.loads(text)

        except json.JSONDecodeError:
            raise ValueError(f"Agent did not return valid JSON:\n{text}")

        except Exception as e:
            # Handle free-tier rate limits (429). The free tier is only
            # 5 requests/minute, so exponential backoff alone isn't reliably
            # enough -- prefer the server's own retryDelay when it gives one.
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                last_error = e
                match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                wait = int(match.group(1)) + 1 if match else 2 ** attempt
                print(f"  [rate limit hit, waiting {wait}s...]")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after 6 attempts (rate limits?): {last_error}")


def _call_openai(system, user, max_tokens):
    client = _get_openai_client()
    last_error = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # GPT-5-series models use max_completion_tokens, not
                # max_tokens, and reasoning tokens are drawn from that same
                # budget -- reasoning_effort="none" keeps hidden reasoning
                # from eating the whole budget before any JSON is emitted
                # (the same failure mode we hit with Gemini's thinking mode).
                max_completion_tokens=max_tokens,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            if resp.usage is not None:
                _record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            text = resp.choices[0].message.content
            text = re.sub(r"```json|```", "", text).strip()
            return json.loads(text)

        except json.JSONDecodeError:
            raise ValueError(f"Agent did not return valid JSON:\n{text}")

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                last_error = e
                wait = 2 ** attempt
                print(f"  [rate limit hit, waiting {wait}s...]")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after 6 attempts (rate limits?): {last_error}")


def call_json(system, user, max_tokens=1000):
    """Call the active provider (LLM_PROVIDER env var, default "gemini")
    and parse its response as JSON.

    Each agent is instructed (via its system prompt) to return ONLY JSON.
    We strip any accidental markdown fences before parsing, and retry on
    rate-limit errors with backoff.
    """
    if PROVIDER == "openai":
        return _call_openai(system, user, max_tokens)
    return _call_gemini(system, user, max_tokens)
