import os
import httpx
from .contracts import Generation, ProviderError

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"


class FireworksProvider:
    def __init__(self, api_key=None, model=None, transport=None, reasoning_effort=None):
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.model = model or os.getenv("FIREWORKS_MODEL") or DEFAULT_MODEL
        self.transport = transport
        self.reasoning_effort = reasoning_effort

    def generate(self, messages, temperature=0.0, max_new_tokens=128, response_schema=None):
        if not self.api_key:
            raise ProviderError("FIREWORKS_API_KEY is not configured on the server")
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_new_tokens, "stream": False}
        if self.reasoning_effort is not None:
            payload['reasoning_effort'] = self.reasoning_effort
        if response_schema is not None:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'RecipeExtraction', 'schema': response_schema}}
        # One bounded request; no automatic retries that might incur duplicate charges.
        try:
            with httpx.Client(timeout=httpx.Timeout(120, connect=10), transport=self.transport) as client:
                response = client.post("https://api.fireworks.ai/inference/v1/chat/completions",
                    headers={"Authorization": "Bearer " + self.api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Expected text response")
                usage = data.get("usage") or {}
                return Generation(raw_output=content, model=data.get("model", self.model),
                    finish_reason=choice.get("finish_reason") or "unknown",
                    input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
        except httpx.HTTPStatusError as exc:
            # Do not expose request headers, API keys or raw upstream error documents.
            status = exc.response.status_code
            if status == 404:
                model_name = self.model.replace(self.api_key, '[redacted]')
                detail = (f"model {model_name!r} is unavailable, not deployed, or not accessible to this account. "
                          "Check FIREWORKS_MODEL (or this feature's model override) in .env, "
                          "then recreate the backend with docker compose up -d backend")
            else:
                detail = {
                    400: "the model rejected the request parameters",
                    401: "authentication failed; check the server's FIREWORKS_API_KEY",
                    402: "check the Fireworks account's billing and usage limits",
                    403: "access was denied; check the API key's permissions",
                    429: "rate limit or deployment capacity exceeded; wait before trying again",
                }.get(status, "the upstream service failed; try again later and check Fireworks service status")
            raise ProviderError(f"Fireworks returned HTTP {status}: {detail}") from exc
        except httpx.RequestError as exc:
            raise ProviderError("Fireworks connection failed or timed out") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Fireworks returned an unexpected response envelope") from exc
