"""
Thin wrapper around the Google Gemini API that returns parsed JSON.

This is a DROP-IN replacement for the Anthropic version: it exposes the same
call_json(system, user) function, so agents.py and orchestrator.py need no
changes at all.

Setup:
    pip install google-genai
    set GEMINI_API_KEY   (Windows)   /   export GEMINI_API_KEY  (Mac/Linux)
"""
import json
import os
import re
import time
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Free-tier model. "gemini-2.5-flash" is no longer available to new API keys
# (Google returns a 404). "gemini-flash-latest" works but its free daily quota
# (20 req/day at time of writing) is easy to exhaust; "gemini-flash-lite-latest"
# is a separate quota bucket with a much higher free-tier allowance.
MODEL = "gemini-flash-lite-latest"


def call_json(system, user, max_tokens=1000):
    """Call Gemini and parse its response as JSON.

    Each agent is instructed (via its system prompt) to return ONLY JSON.
    We strip any accidental markdown fences before parsing, and retry once
    on a rate-limit (429) error with a short backoff.
    """
    last_error = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=MODEL,
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
