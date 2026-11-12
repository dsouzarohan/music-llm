# learnings

Plain-language notes on concepts met while building this project, one file per concept. Each
one says what the idea is, why it exists, a simple picture of how it works, and where it lives in
our code. They're for remembering, not reference docs: `docs/architecture/` describes how the
code works today.

| Note | In one line |
|---|---|
| [RoPE](rope.md) | Rotate q and k by position so attention scores depend only on distance |
| [KV cache](kv-cache.md) | Keep old tokens' keys and values so generation processes one token per step |
| [MIDI tokenization](midi-tokenization.md) | MIDI messages, channels, and how time works in MIDI |
