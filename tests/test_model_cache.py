import torch
from draftlab.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=50, block_size=64, n_layer=2, n_head=4, n_embd=32)


def make_model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def rand_ids(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CFG.vocab_size, (1, n), generator=g)


@torch.no_grad()
def test_one_token_at_a_time_matches_full_forward():
    m, ids = make_model(), rand_ids(20)
    full = m(ids)
    cache = m.new_cache()
    steps = torch.cat([m(ids[:, t:t + 1], cache) for t in range(ids.shape[1])], dim=1)
    assert torch.allclose(full, steps, atol=1e-5)


@torch.no_grad()
def test_multi_token_chunk_matches_full_forward():
    m, ids = make_model(), rand_ids(24)
    full = m(ids)
    cache = m.new_cache()
    m(ids[:, :10], cache)                   # prefill
    chunk_logits = m(ids[:, 10:15], cache)  # k = 5 tokens in ONE pass
    assert cache.length == 15
    assert torch.allclose(full[:, 10:15], chunk_logits, atol=1e-5)


@torch.no_grad()
def test_rollback_then_continue_matches_fresh_run():
    m = make_model()
    prefix, wrong, right = rand_ids(10, 1), rand_ids(5, 2), rand_ids(4, 3)

    cache = m.new_cache()
    m(prefix, cache)
    m(wrong, cache)             # draft's guesses, cache now length 15
    cache.rollback(12)          # accept 2 of 5 guesses
    assert cache.length == 12
    out = m(right, cache)       # continue from the accepted point

    ref_ids = torch.cat([prefix, wrong[:, :2], right], dim=1)
    ref = m(ref_ids)[:, 12:]
    assert torch.allclose(ref, out, atol=1e-5)


@torch.no_grad()
def test_rollback_to_zero_and_reuse():
    m, ids = make_model(), rand_ids(8)
    cache = m.new_cache()
    m(rand_ids(8, 9), cache)
    cache.rollback(0)
    assert torch.allclose(m(ids, cache), m(ids), atol=1e-5)