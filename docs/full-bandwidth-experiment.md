# Full-bandwidth Transformer experiment

`docs/current-harness-pipeline.mmd` records the existing text-mediated
pipeline. `docs/full-bandwidth-transformer-target.mmd` records the desired
architecture; blue nodes are implemented in this branch and gray nodes require
a feedback-capable model runtime and trained weights.

## What this branch implements

The Harness now has an opt-in `--full-bandwidth-feedback` mode. It asks the
selected backend whether the model advertises this capability and fails closed
when it does not. A compatible backend receives this extension on every
completion request:

```json
{
  "full_bandwidth_feedback": {
    "enabled": true,
    "protocol_version": 1
  }
}
```

The option is forwarded to annotation, thought-process, and response
completions. This is deliberately an opaque server-side protocol: the Harness
must never copy a hidden-state vector through JSON, storage, or prompt tokens.

## Required model-server work

A compatible server must:

1. Advertise `capabilities.full_bandwidth_feedback: true` from its model
   properties endpoint.
2. Run a model trained with the Layer-N-to-Layer-0 gated feedback path.
3. Retain one previous final hidden state per active decoding sequence next to
   its KV cache, fuse it into the next token's input, and reset it when that
   sequence resets.
4. Ignore neither the request flag nor its version: reject unsupported protocol
   versions explicitly.

Stock llama.cpp and ordinary Gemma GGUF weights do not meet these requirements.
They remain supported in the default mode, where this option is absent.
