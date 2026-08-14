from Harness.history import format_conversation_history


def test_history_includes_prior_messages_but_not_current_turn():
    history = format_conversation_history(
        [
            {"turn_id": "one", "role": "user", "content": "Earlier question"},
            {"turn_id": "one", "role": "assistant", "content": "Earlier answer"},
            {"turn_id": "two", "role": "user", "content": "Current question"},
        ],
        current_turn_id="two",
    )

    assert history == (
        "[Conversation history]\n"
        "User: Earlier question\n\n"
        "Assistant: Earlier answer"
    )
