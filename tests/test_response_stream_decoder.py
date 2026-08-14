from Harness.structured_output import JSONFieldStreamDecoder


def test_json_field_stream_decoder_emits_string_value_as_json_arrives():
    decoder = JSONFieldStreamDecoder("response")

    assert decoder.feed('{"response":"Hello') == "Hello"
    assert decoder.feed('\\nworld') == "\nworld"
    assert decoder.feed('!"}') == "!"
    assert decoder.result() == "Hello\nworld!"
