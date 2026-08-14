from Harness.responder import _ResponseStreamDecoder


def test_response_stream_decoder_emits_string_value_as_json_arrives():
    decoder = _ResponseStreamDecoder()

    assert decoder.feed('{"response":"Hello') == "Hello"
    assert decoder.feed('\\nworld') == "\nworld"
    assert decoder.feed('!"}') == "!"
    assert decoder.result() == "Hello\nworld!"
