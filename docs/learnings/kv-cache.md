# KV cache

**One line:** when generating, keep every token's key and value vectors, so each new step only
has to process the newest token.

Code: `KVCache` and `generate(use_cache=True)` in `src/model/model.py`.

## The problem

Generation makes one token at a time. Without a cache:

- step 101 runs the model over tokens 1–100
- step 102 runs it over tokens 1–101
- step 103 runs it over tokens 1–102 …

Tokens 1–100 get processed again and again, even though nothing about them changed.

## What attention needs from old tokens

In each layer every token makes three vectors:

- **q (query):** what am I looking for?
- **k (key):** what do I contain? (others compare their q against this)
- **v (value):** what do I hand over if someone attends to me?

The new token compares **its q** with **every earlier k**, then mixes **every earlier v**.

**Key point:** the model is causal, so token 50's k and v depend only on tokens 1–50. A new token
at 101 can't change them. Once computed, they're final, so we can just keep them.

## How the cache works

1. **Prefill:** run the whole prompt once and store each layer's k and v.
2. **Each step:** run only the new token. Compute its q, k, v, append k and v to the cache,
   and attend with q over all cached keys and values.

```python
logits, _ = self(idx[:, -self.block_size:], cache=cache)   # first: whole prompt
logits, _ = self(idx[:, -1:], cache=cache)                  # then: just the newest token
```

**An analogy:** reading a book with notes. To understand page 101, you read page 101 and check
your notes. You don't reread pages 1–100.

## Why not a "QKV" cache?

An old token's q was only used to compute *that token's* output, which is already done. Only
the new token asks questions, so only its q is needed.

## Position with a cache

The new token sits at position `cache.length`.

- **RoPE:** rotate the new q and k by that position. Cached keys were already rotated by
  their own positions, and nothing needs updating.
- **Huang:** the new token's distances to the cache are 0, −1, −2 …, which map straight to
  rows of `rel_bias`. There's no skew, since there's only one query row.

## Costs and limits

- **Memory:** layers × 2 (k and v) × positions × width × bytes. At the 186M shape with a full
  2048 window in bf16, that's about 150 MB per sequence. Fine for one sample; it's the big
  memory cost when serving large batches.
- **Training doesn't use it.** Training processes all positions in one parallel pass, so
  there's nothing repeated to skip.
- **The 2048 boundary:** when the window is full, the oldest token has to drop out. But every
  cached k/v was computed while that token was still visible (token 2's layer-5 key was built
  from layers that saw token 1). So the cache is thrown away and the last 2048 tokens are
  reprocessed. Output stays identical to no-cache, but past 2048 there's no speedup.
  Our 64 + 512-token samples never get there.

## Numbers

186M-shape model (random weights) on the Mac (MPS), 64-token prompt + 256 new tokens:

| | no cache | cache |
|---|---|---|
| Huang | 8.5 s | 2.6 s |
| RoPE | 6.1 s | 2.2 s |

The gap grows with length. What's left (~9 ms/token) is mostly the overhead of launching a
few hundred small GPU kernels per step at batch size 1, not the maths itself.
