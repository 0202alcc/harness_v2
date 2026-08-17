import pytest

from Harness.full_bandwidth import FullBandwidthFeedback
from Harness.responder import Responder
from Harness.thought_processor import ThoughtProcessor
from storage import ChatStorage


class FeedbackCapableLLM:
    def __init__(self, supported: bool):
        self.supported = supported

    def supports_full_bandwidth(self, model: str) -> bool:
        assert model == "feedback-model"
        return self.supported


def test_disabled_feedback_adds_no_completion_options():
    assert FullBandwidthFeedback().completion_options(
        FeedbackCapableLLM(supported=False),
        "feedback-model",
    ) == {}


def test_enabled_feedback_requires_a_feedback_capable_model():
    with pytest.raises(RuntimeError, match="does not advertise"):
        FullBandwidthFeedback(enabled=True).completion_options(
            FeedbackCapableLLM(supported=False),
            "feedback-model",
        )


def test_enabled_feedback_sends_the_versioned_server_protocol():
    assert FullBandwidthFeedback(enabled=True).completion_options(
        FeedbackCapableLLM(supported=True),
        "feedback-model",
    ) == {
        "full_bandwidth_feedback": {
            "enabled": True,
            "protocol_version": 1,
        }
    }


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def tokenize(self, text, **kwargs):
        return [1, 2, 3]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def stream_complete(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            yield {"content": '{"chunk":"Thought","done":true}', "tokens": [4]}
        else:
            yield {"content": '{"chunk":"Answer","done":true}', "tokens": [5]}


def test_visible_synthesis_and_response_forward_feedback_protocol(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="feedback-model")
    llm = RecordingLLM()
    options = FullBandwidthFeedback(enabled=True).completion_options(
        FeedbackCapableLLM(supported=True),
        "feedback-model",
    )

    ThoughtProcessor(
        llm=llm, store=store, model="feedback-model", user_id="user",
        chat_id="chat", instruction="Think.", completion_options=options,
    ).generate(
        message="Question", annotations=[], system_prompt=None,
        conversation_history=None, run_id="run", turn_id="turn",
    )
    Responder(
        llm=llm, store=store, model="feedback-model", user_id="user",
        chat_id="chat", instruction="Answer.", completion_options=options,
    ).generate(
        message="Question", thought_process="Thought", system_prompt=None,
        conversation_history=None, run_id="run", turn_id="turn",
    )

    assert [call["full_bandwidth_feedback"] for call in llm.calls] == [
        options["full_bandwidth_feedback"],
        options["full_bandwidth_feedback"],
    ]
