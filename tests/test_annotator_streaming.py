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
        self.completion_kwargs = []
        self.health_checks = 0

    def tokenize(self, text, **kwargs):
        return [len(text)]

    def stream_complete(self, *, prompt, **kwargs):
        self.prompts.append(list(prompt))
        self.completion_kwargs.append(kwargs)
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        if callable(action):
            return action()
        return iter(action)

    def health(self):
        self.health_checks += 1
        return {"ok": True}


def stream_event(content, token, *, final=False, cache_n=0, prompt_n=4, **metadata):
    return {
        "content": content,
        "tokens": [] if final else [token],
        "id_slot": 0,
        "stop": final,
        "timings": {
            "cache_n": cache_n,
            "prompt_n": prompt_n,
        },
        **metadata,
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
            stream_event('{"chunk":"Hello","done":true}', 10),
            stream_event("", 11, final=True),
        ]],
    )

    result = run_annotation(annotator)

    assert result["annotation"]["text"] == "Hello"
    assert result["annotation"]["token_ids"] == [10]
    assert result["thinking_token_ids"] == [1, 2, 4, 3, len("Hello")]
    assert llm.prompts == [[1, 2, 4, 3]]

    [record] = read_trace(store)
    assert record["status"] == "success"
    assert record["request"]["attempt"]["kind"] == "initial"
    assert record["response"]["tokens_cached"] == 0
    assert record["response"]["tokens_evaluated"] == 4


def test_next_chunk_uses_compact_marker_text_not_generated_json_tokens(tmp_path):
    annotator, llm, _store = make_annotator(
        tmp_path,
        [
            [stream_event('{"chunk":"Hello","done":true}', 10)],
            [stream_event('{"chunk":"Again","done":true}', 11)],
        ],
    )

    first = run_annotation(annotator)
    annotator.annotate(
        thinking_token_ids=first["thinking_token_ids"],
        chunk={"index": 1, "token_ids": [4], "text": "source"},
        run_id="run",
        turn_id="turn",
    )

    assert llm.prompts[1] == [1, 2, 4, 3, len("Hello"), 2, 4, 3]
    assert 10 not in llm.prompts[1]


def test_annotation_forwards_full_bandwidth_protocol_to_the_backend(tmp_path):
    annotator, llm, _store = make_annotator(
        tmp_path,
        [[stream_event('{"chunk":"Hello","done":true}', 10)]],
    )
    annotator.completion_options = {
        "full_bandwidth_feedback": {
            "enabled": True,
            "protocol_version": 1,
        }
    }

    run_annotation(annotator)

    # A production backend receives this opaque flag and owns the hidden state.
    assert llm.completion_kwargs[0]["full_bandwidth_feedback"] == {
        "enabled": True,
        "protocol_version": 1,
    }


def test_connection_drop_before_tokens_retries_original_prompt(tmp_path):
    annotator, llm, store = make_annotator(
        tmp_path,
        [
            ProviderError("dropped", retryable=True),
            [
                stream_event('{"chunk":"Recovered","done":true}', 10),
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


def test_mid_stream_drop_retries_only_the_current_json_envelope(tmp_path):
    def interrupted_stream():
        yield stream_event('{"chunk":"First', 10)
        raise ProviderError("dropped", retryable=True)

    annotator, llm, store = make_annotator(
        tmp_path,
        [
            interrupted_stream,
            [
                stream_event('{"chunk":"First second","done":true}', 11),
                stream_event("", 0, final=True),
            ],
        ],
    )

    with patch("Harness.annotator.time.sleep"):
        result = run_annotation(annotator)

    assert result["annotation"]["text"] == "First second"
    assert result["annotation"]["token_ids"] == [11]
    assert llm.prompts == [[1, 2, 4, 3], [1, 2, 4, 3]]
    assert llm.health_checks == 0

    failed, succeeded = read_trace(store)
    assert failed["response"]["generated_token_ids"] == [10]
    assert succeeded["request"]["attempt"]["kind"] == "initial_retry"
    assert succeeded["request"]["json_schema"]["required"] == ["chunk", "done"]


def test_annotation_continues_after_a_complete_unfinished_envelope(tmp_path):
    annotator, llm, _store = make_annotator(
        tmp_path,
        [
            [stream_event('{"chunk":"First ","done":false}', 10)],
            [stream_event('{"chunk":"second","done":true}', 11)],
        ],
    )

    result = run_annotation(annotator)

    assert result["annotation"]["text"] == "First second"
    assert result["annotation"]["token_ids"] == [10, 11]
    assert len(llm.prompts) == 2


def test_incomplete_normal_stream_is_traced_as_invalid_json_then_retried(tmp_path):
    annotator, _llm, store = make_annotator(
        tmp_path,
        [
            [stream_event('{"chunk":"partial', 10)],
            [stream_event('{"chunk":"recovered","done":true}', 11)],
        ],
    )

    with patch("Harness.annotator.time.sleep"):
        result = run_annotation(annotator)

    assert result["annotation"]["text"] == "recovered"
    failed, succeeded = read_trace(store)
    assert failed["status"] == "error"
    assert failed["response"]["terminal_condition"] == "invalid_json"
    assert succeeded["status"] == "success"


def test_annotation_token_cutoff_retries_current_envelope_with_more_budget(tmp_path):
    annotator, llm, store = make_annotator(
        tmp_path,
        [
            [stream_event('{"chunk":"partial', 10, truncated=True, stop_type="token_limit")],
            [stream_event('{"chunk":"recovered","done":true}', 11)],
        ],
    )

    with patch("Harness.annotator.time.sleep"):
        result = run_annotation(annotator)

    assert result["annotation"]["text"] == "recovered"
    assert llm.completion_kwargs[0]["n_predict"] == 4
    assert llm.completion_kwargs[1]["n_predict"] == 20
    failed, _succeeded = read_trace(store)
    assert failed["response"]["terminal_condition"] == "token_limit"
