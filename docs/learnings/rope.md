# RoPE (rotary position embeddings)

**One line:** rotate q and k by an angle that depends on the token's position, so that their
dot product only depends on how far apart the two tokens are.

Code: `rope_tables` and `apply_rope` in `src/model/model.py`, used when `pos = "rope"`.

## The problem

Attention compares a query with keys (`q · k`). On its own that ignores order, so the model
needs some notion of position. We had two:

- **Absolute position embedding:** add a learned "I'm slot 412" vector to each token.
- **Huang relative bias:** add a second score `q · E[distance]`, where `E` holds one learned
  vector per distance (`rel_bias`). The skew trick lines those distances up with keys.

Both work, but they cost a lot. `rel_bias` was 37.7M parameters (20% of the 186M model),
and the extra score matrix blocks SDPA's fast flash kernel.

## The trick: rotation

Take a 2-D vector and rotate it. Its length stays the same; only its angle changes.

- Rotate the query at position m by `m·θ`.
- Rotate the key at position n by `n·θ`.
- A dot product depends on the angle **between** the two vectors, which changed by
  `(m − n)·θ`. The absolute positions cancel; only the distance is left.

```
(R(mθ) q) · (R(nθ) k)  =  qᵀ R((n−m)θ) k
```

**An analogy:** two people on a merry-go-round stand 90° apart. However far it spins, they're still
90° apart.

**Check:** the same q and k placed at positions (5, 2), (105, 102) and (2000, 1997) all give
exactly the same score, because all three pairs are 3 apart.

## Why many speeds: clock hands

A head has 64 dims, so 32 pairs, each a 2-D plane spinning at its own speed
`θ_i = 10000^(−2i/64)`.

One speed alone wraps around: at 30° per token, distance 12 looks like distance 0. So:

- **fast pairs** (second hand): repeat every ~6 tokens. They tell distance 1 from 2.
- **slow pairs** (hour hand): repeat every ~47,000 tokens. They tell near from far.

Together they pin down any distance, the same way a clock's hands together give the exact time.

## In code

```python
inv_freq = 10000 ** (-torch.arange(0, Dh, 2) / Dh)   # speed of each pair
angles = torch.outer(torch.arange(T), inv_freq)       # position × speed
x1, x2 = x.chunk(2, dim=-1)                           # the two halves of each pair
out = cat([x1*cos - x2*sin, x1*sin + x2*cos])         # 2-D rotation
```

- It's applied to **q and k only**, not v. Position decides *who attends to whom*; v is the
  content being passed along.
- A rotation doesn't change length, so the `1/√Dh` scaling stays the same.
- It has **zero learned parameters** and there's no position embedding any more.
- There's no extra score matrix, so SDPA can use flash attention.

## RoPE vs Huang, in short

| | Huang | RoPE |
|---|---|---|
| Distance enters as | an extra **added** score `q · E[d]` | a rotation **inside** `q · k` |
| Parameters | `rel_bias` (big) + position embedding | none |
| Flash attention | no (needs a bias matrix) | yes |
| Max length | hard limit `block_size` | none in principle, but quality drops past the trained length |

## RoPE and music tokens

- RoPE knows **token index** ("8 tokens back"), not **musical time** ("beat 2 of bar 17").
  In REMI, musical time comes from the `Bar` and `Position_x` tokens. The model works out the
  beat by attending to them.
- Token distance is a poor stand-in for musical time. A dense chord bar has many tokens, a
  rest bar has few. BPE merges blur token types even more.
- Every token gets rotated, velocity included. That's fine: RoPE only changes the attention
  *scores*, never the token's content.
- The model can choose how much position matters, per head and per token. It does that by
  putting q/k energy into fast pairs (positional: "the token right before me") or slow pairs
  (content: "any earlier `Pitch_81`"). This happens in the learned q/k projections, not in the
  embedding. Phase 2 T5 has a probe to check this.

## Later idea (Phase 4): rotate by musical attributes

Instead of token index, rotate some pairs by bar, some by beat, maybe some by **pitch**. Rotate
a pair by `pitch × 360°/12`, and C4 and C5 land on the same angle (octave equivalence for
free), while a fifth is always the same angle in every key. Vision models already rotate
different pairs by different axes (e.g. Qwen2-VL's M-RoPE). The catch: non-note tokens have no
pitch, and BPE merges mix attributes, so it needs tokenizer work first.

## Read more

- Su et al., 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Barbero et al., 2024, *Round and Round We Go! What makes Rotary Positional Encodings useful?*
