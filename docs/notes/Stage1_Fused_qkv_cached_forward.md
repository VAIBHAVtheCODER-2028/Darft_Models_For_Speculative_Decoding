# Stage 1: Fused-QKV model with multi-token cached forward and rollback

## Why this stage exists
Speculative decoding needs the target model to score k guessed tokens in ONE pass,
and to undo guesses that get rejected. My old `kv_cache.py` could only add one token
at a time and could not undo. This stage upgrades the model so it can do both.

## Concepts

### 1. Fused QKV
- Before: a Python loop over heads, each head doing its own Q, K, V projections.
- Now: ONE `nn.Linear(n_embd, 3 * n_embd)` gives Q, K, V for all heads together.
  `split(C, dim=2)` separates them, then `view(B, T, n_head, head_dim).transpose(1, 2)`
  puts heads in a batch dimension, so every head runs in one batched operation.
- Same maths, fewer kernel launches. It also removes the loop that limited my earlier
  KV-cache benchmark, so later speedup numbers mean more.

### 2. Multi-token cached forward
- `forward(idx, cache)` takes only the NEW chunk (T tokens, T can be > 1).
- Positions continue from `cache.length`, so the chunk gets positions past..past+T-1.
- It returns logits for EVERY position in the chunk, not just the last one.
  The target needs all k+1 distributions to verify the draft's guesses.
- K and V of the chunk are written into the cache, then attention reads past + chunk.

### 3. The mask (the hard part)
Query i of the chunk sits at absolute position `past + i`. It may see key j if `j <= past + i`.
- All `past` cached keys: visible to every query.
- Inside the chunk: causal triangle.

Example: past = 2, T = 3, so S = 5 keys (columns), 3 queries (rows). 1 = visible.

```
            key: 0 1 | 2 3 4      (0,1 = cache, 2,3,4 = new chunk)
query 0 (pos 2):  1 1 | 1 0 0
query 1 (pos 3):  1 1 | 1 1 0
query 2 (pos 4):  1 1 | 1 1 1
```

In code: `torch.ones(T, S).tril(diagonal=past)`.
With `past = 0` this is the ordinary causal mask; with `T = 1` it is a row of all ones.

### 4. Rollback
- `cache.rollback(n)` sets `cache.length = n`. Nothing is erased.
- Safe because attention only reads `[:past + T]`, and the next write overwrites
  everything beyond `length`.
- Used when the target rejects a guess: the cache is cut back to the last accepted token.

## Design decisions
- Cache is preallocated `(n_layer, B, n_head, block_size, head_dim)`: no concatenation, no reallocation.
- `cache.length` is advanced in `GPT.forward` AFTER all layers run. Every layer must write
  at the same `past`, so updating it per layer would be a bug.
- `scaled_dot_product_attention` with an explicit boolean mask, so the same code path
  serves the full pass, one token, and a k-token chunk.
- Weight tying (head shares token embedding), no biases in linears, learned positions, pre-norm.

## What the tests prove (tests/test_model_cache.py)
1. One token at a time with cache == full forward.
2. Prefill 10 tokens, then a 5-token chunk in one pass == full forward on those 5 positions.
3. Feed 5 guesses, rollback to accept 2, continue == fresh run on prefix + 2 accepted + new.
4. Rollback to 0 and reuse == fresh run.
Sanity check: changing the mask to a plain causal triangle makes tests 1-3 fail.

## Things to remember for later stages
- Stage 2 will call: target forward on [last_token + k draft tokens] -> k+1 distributions,
  then `rollback(prefix_len + n_accepted)`.
- The draft also keeps its own cache and needs rollback too.
- Cache stays in sync only if both models roll back by the same accepted count.
- Max sequence is `block_size`: speculative chunks must not push past it.

## Open questions / my notes
-If you want to be sure, change tril(diagonal=past) to tril(diagonal=0) once and watch tests 1-3 fail. If you can say why they fail, you’ve got it.(Claude's opinion)