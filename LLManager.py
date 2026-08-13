from __future__ import annotations
from typing import Any, Sequence
import httpx

class ProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""

class Provider:
    """
    Base interface for an LLM backend.

    Provider implementations should hide backend-specific HTTP/API details
    from the Harness.
    """

    def __init__(
        self,
        *,  base_url: str, 
        api_key: str | None = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def close(self) -> None:
        pass


class LlamaCppProvider(Provider):
    """
    Native llama.cpp server provider.

    Uses llama.cpp's native endpoints rather than relying exclusively on the
    OpenAI-compatible /v1 API.
    """

    def __init__(
        self,
        *, base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
        )

        # Existing config may use:
        #
        #     http://host:10539/v1
        #
        # but native llama.cpp endpoints live at:
        #
        #     http://host:10539/tokenize
        #     http://host:10539/completion
        #     ...
        #
        # So normalize back to the llama-server root.
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3].rstrip("/")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        try:
            response = self.client.request(
                method,
                path,
                params=params,
                json=json,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Could not reach llama.cpp server at "
                f"{self.base_url}{path}: {exc}"
            ) from exc

        if response.is_error:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text

            raise ProviderError(
                f"llama.cpp request failed "
                f"({response.status_code} {method} {path}): "
                f"{detail}"
            )

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                f"llama.cpp returned non-JSON data for "
                f"{method} {path}: {response.text[:500]}"
            ) from exc

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            path,
            params=params,
        )

    def _post(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            path,
            json=payload,
        )

    # ------------------------------------------------------------------
    # Server / model management
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Check whether llama-server is ready.

        llama.cpp returns:
            200 -> {"status": "ok"}
            503 -> model is still loading
        """

        try:
            response = self.client.get("/health")
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "status": "unreachable",
                "error": str(exc),
            }

        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}

        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            **data,
        }

    def list_models(
        self,
        *,
        reload: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return all models known to the llama.cpp router.
        """

        params = {}

        if reload:
            params["reload"] = 1

        data = self._get(
            "/models",
            params=params or None,
        )

        return data.get("data", [])

    def load_model(
        self,
        model: str,
    ) -> dict[str, Any]:
        """
        Explicitly load a model.
        """

        return self._post(
            "/models/load",
            payload={
                "model": model,
            },
        )

    def unload_model(
        self,
        model: str,
    ) -> dict[str, Any]:
        """
        Explicitly unload a model.
        """

        return self._post(
            "/models/unload",
            payload={
                "model": model,
            },
        )

    def model_props(
        self,
        model: str,
        *,
        autoload: bool = False,
    ) -> dict[str, Any]:
        """
        Get runtime properties for one model.

        Useful fields include:
            default_generation_settings
            total_slots
            model_path
            chat_template
            chat_template_caps
            modalities
            build_info
            is_sleeping
        """

        return self._get(
            "/props",
            params={
                "model": model,
                "autoload": str(autoload).lower(),
            },
        )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
        model: str,
        *,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = False,
    ) -> list[int] | list[dict[str, Any]]:
        """
        Tokenize text using the tokenizer belonging to the actual
        llama.cpp-loaded model.

        For normal Harness chunking:
            add_special=False
            with_pieces=False
        """

        data = self._post(
            "/tokenize",
            payload={
                "model": model,
                "content": text,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": with_pieces,
            },
        )

        return data["tokens"]

    def detokenize(
        self,
        token_ids: Sequence[int],
        model: str,
    ) -> str:
        """
        Convert model token IDs back to text.
        """

        data = self._post(
            "/detokenize",
            payload={
                "model": model,
                "tokens": list(token_ids),
            },
        )

        return data["content"]

    # ------------------------------------------------------------------
    # Chat-template handling
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        """
        Apply the model's native chat template without performing inference.

        Returns the fully formatted prompt string.
        """

        data = self._post(
            "/apply-template",
            payload={
                "model": model,
                "messages": messages,
            },
        )

        return data["prompt"]

    # ------------------------------------------------------------------
    # Low-level inference
    # ------------------------------------------------------------------

    def prefill(
        self,
        prompt: str | Sequence[int],
        model: str,
        *,
        cache_prompt: bool = True,
        id_slot: int | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a prompt into the KV cache without generating new tokens.

        llama.cpp implements this through /completion with n_predict=0.
        """

        return self.complete(
            prompt=prompt,
            model=model,
            n_predict=0,
            cache_prompt=cache_prompt,
            return_tokens=False,
            id_slot=id_slot,
        )

    def complete(
        self,
        prompt: str | Sequence[int],
        model: str,
        *,
        n_predict: int = 128,
        cache_prompt: bool = True,
        return_tokens: bool = True,
        id_slot: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        grammar: str | None = None,
        json_schema: dict[str, Any] | None = None,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """
        Low-level native llama.cpp completion.

        `prompt` may be either:
            - text
            - raw model token IDs

        Returns llama.cpp's full response so the Harness can inspect:
            content
            tokens
            timings
            tokens_cached
            tokens_evaluated
            stop_type
            truncated
            generation_settings
            etc.
        """

        if extra_params.get("stream"):
            raise NotImplementedError(
                "Streaming is intentionally not implemented in "
                "LLManager.complete() yet."
            )

        if isinstance(prompt, str):
            normalized_prompt: str | list[int] = prompt
        else:
            normalized_prompt = list(prompt)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": normalized_prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "return_tokens": return_tokens,
            "stream": False,
        }

        if id_slot is not None:
            payload["id_slot"] = id_slot

        if temperature is not None:
            payload["temperature"] = temperature

        if top_k is not None:
            payload["top_k"] = top_k

        if top_p is not None:
            payload["top_p"] = top_p

        if min_p is not None:
            payload["min_p"] = min_p

        if seed is not None:
            payload["seed"] = seed

        if stop is not None:
            payload["stop"] = stop

        if grammar is not None:
            payload["grammar"] = grammar

        if json_schema is not None:
            payload["json_schema"] = json_schema

        payload.update(extra_params)

        return self._post(
            "/completion",
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> LlamaCppProvider:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class LLManager:
    """
    Harness-facing model manager.

    The Harness talks to this class rather than directly talking to
    llama.cpp.
    """

    def __init__(
        self,
        provider: Provider,
    ):
        self.provider = provider

    def health(self):
        return self.provider.health()

    def list_models(self, **kwargs):
        return self.provider.list_models(**kwargs)

    def load_model(self, model: str):
        return self.provider.load_model(model)

    def unload_model(self, model: str):
        return self.provider.unload_model(model)

    def model_props(self, model: str, **kwargs):
        return self.provider.model_props(model, **kwargs)

    def tokenize(self, text: str, model: str, **kwargs):
        return self.provider.tokenize(text, model, **kwargs)

    def detokenize(self, token_ids: Sequence[int], model: str):
        return self.provider.detokenize(token_ids, model)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ):
        return self.provider.apply_chat_template(
            messages,
            model,
        )

    def prefill(self, prompt, model: str, **kwargs):
        return self.provider.prefill(
            prompt,
            model,
            **kwargs,
        )

    def complete(self, prompt, model: str, **kwargs):
        return self.provider.complete(
            prompt,
            model,
            **kwargs,
        )

    def close(self):
        self.provider.close()