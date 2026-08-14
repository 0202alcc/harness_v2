import json
from pathlib import Path
from unittest.mock import patch

from Harness.annotator import Annotator
from LLManager import ProviderError
from storage import ChatStorage


class FakeLLM:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []
        self.health_checks = 0

    def tokenize(self, text, **kwargs):
        return [len(text)]

    def stream_complete(self, *, prompt, **kwargs):
        self.prompts.append(list(prompt))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        if callable(action):
            return action()
        return iter(action)

    def health(self):
        self.health_checks += 1
        return {"ok": True}


def stream_event(content, token, *, final=False, cache_n=0, prompt_n=4):
    return {
        "content": content,
        "tokens": [] if final else [token],
        "id_slot": 0,
        "stop": final,
        "timings": {
            "cache_n": cache_n,
            "prompt_n": prompt_n,
        },
    }


def make_annotator(tmp_path, actions):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")
    llm = FakeLLM(actions)
    annotator = Annotator(
        llm=llm,
        store=store,
        model="fake",
        user_id="user",
        chat_id="chat",
        instruction="Annotate.",
        n_predict=4,
    )
    annotator._source_marker_tokens = [2]
    annotator._annotation_marker_tokens = [3]
    return annotator, llm, store


def run_annotation(annotator):
    return annotator.annotate(
        thinking_token_ids=[1],
        chunk={"index": 0, "token_ids": [4], "text": "source"},
        run_id="run",
        turn_id="turn",
    )


def read_trace(store):
    path = Path(store.get_llama_io_path("user", "chat"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_normal_streaming_persists_one_successful_completion(tmp_path):
    annotator, llm, store = make_annotator(
        tmp_path,
        [[
            stream_event("Hello", 10),
            stream_event("", 11, final=True),
        ]],
    )

    result = run_annotation(annotator)

    assert result["annotation"]["text"] == "Hello"
    assert result["annotation"]["token_ids"] == [10]
    assert llm.prompts == [[1, 2, 4, 3]]

    [record] = read_trace(store)
    assert record["status"] == "success"
    assert record["request"]["attempt"]["kind"] == "initial"
    assert record["response"]["tokens_cached"] == 0
    assert record["response"]["tokens_evaluated"] == 4


def test_connection_drop_before_tokens_retries_original_prompt(tmp_path):
    annotator, llm, store = make_annotator(
        tmp_path,
        [
            ProviderError("dropped", retryable=True),
            [
                stream_event("Recovered", 10),
                stream_event("", 0, final=True),
            ],
        ],
    )

    with patch("Harness.annotator.time.sleep"):
        result = run_annotation(annotator)

    assert result["annotation"]["token_ids"] == [10]
    assert llm.prompts == [[1, 2, 4, 3], [1, 2, 4, 3]]

    failed, succeeded = read_trace(store)
    assert failed["status"] == "error"
    assert succeeded["status"] == "success"
    assert succeeded["request"]["attempt"]["kind"] == "initial_retry"


def test_mid_stream_drop_continues_from_received_token_prefix(tmp_path):
    def interrupted_stream():
        yield stream_event("First", 10)
        raise ProviderError("dropped", retryable=True)

    annotator, llm, store = make_annotator(
        tmp_path,
        [
            interrupted_stream,
            [
                stream_event(" second", 11),
                stream_event("", 0, final=True),
            ],
        ],
    )

    with patch("Harness.annotator.time.sleep"):
        result = run_annotation(annotator)

    assert result["annotation"]["text"] == "First second"
    assert result["annotation"]["token_ids"] == [10, 11]
    assert llm.prompts == [[1, 2, 4, 3], [1, 2, 4, 3, 10]]
    assert llm.health_checks == 1

    failed, succeeded = read_trace(store)
    assert failed["response"]["generated_token_ids"] == [10]
    assert succeeded["request"]["attempt"]["kind"] == "continuation"
