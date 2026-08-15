from Harness.harness import Harness
from storage import ChatStorage


class FakeLLM:
    def tokenize(self, text, **kwargs):
        return [len(text)]


def test_harness_reserves_enough_tokens_for_structured_annotations(tmp_path):
    store = ChatStorage(tmp_path)
    store.create_chat("chat", "user", model="fake")

    harness = Harness(
        llm=FakeLLM(), store=store, model="fake", user_id="user", chat_id="chat",
        annotation_instruction="Annotate.", thought_process_instruction="Think.",
        response_instruction="Respond.",
    )

    assert harness.annotator.n_predict == 96
