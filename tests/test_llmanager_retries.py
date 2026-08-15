from unittest.mock import patch

from LLManager import LLManager, ProviderError


class FakeProvider:
    def __init__(self):
        self.template_calls = 0
        self.stream_calls = 0

    def apply_chat_template(self, messages, model):
        self.template_calls += 1
        if self.template_calls == 1:
            raise ProviderError("dropped", retryable=True)
        return "formatted"

    def stream_complete(self, prompt, model, **kwargs):
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise ProviderError("dropped", retryable=True)
        yield {"content": "ok", "tokens": [1]}


def test_safe_setup_calls_retry_retryable_disconnects():
    provider = FakeProvider()
    manager = LLManager(provider=provider)

    with patch("LLManager.time.sleep"):
        assert manager.apply_chat_template([], "fake") == "formatted"

    assert provider.template_calls == 2


def test_stream_retries_only_before_first_output_token():
    provider = FakeProvider()
    manager = LLManager(provider=provider)

    with patch("LLManager.time.sleep"):
        events = list(manager.stream_complete([], "fake", retry_before_first_token=True))

    assert events == [{"content": "ok", "tokens": [1]}]
    assert provider.stream_calls == 2
