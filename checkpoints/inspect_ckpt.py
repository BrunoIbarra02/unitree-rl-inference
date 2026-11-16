import torch, sys

def inspect(path, obs_dim):
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    print("--- named_buffers ---")
    for name, buf in m.named_buffers():
        print(f"  {name}: shape={tuple(buf.shape)} dtype={buf.dtype}")
    print("--- named_parameters (count only) ---")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  total params: {n_params}")
    print("--- methods ---")
    try:
        print(" ", [m_ for m_ in dir(m) if not m_.startswith('_')][:40])
    except Exception as e:
        print("  (couldn't list)", e)
    print("--- has reset_memory? ---", hasattr(m, "reset_memory"))
    print("--- forward code (truncated) ---")
    try:
        code = m.code
        print(code[:2000])
    except Exception as e:
        print("  no .code:", e)

    print(f"--- test forward with zeros({obs_dim}) ---")
    with torch.inference_mode():
        x = torch.zeros(1, obs_dim) if _expects_batch(m, obs_dim) else torch.zeros(obs_dim)
        try:
            out = m(x)
        except Exception as e:
            print("  batched call failed:", e)
            x = torch.zeros(obs_dim)
            out = m(x)
        if isinstance(out, tuple):
            print(f"  output is TUPLE of len {len(out)}")
            for i, o in enumerate(out):
                _describe(o, i)
        else:
            _describe(out, 0)

def _expects_batch(m, obs_dim):
    return True

def _describe(o, i):
    if torch.is_tensor(o):
        print(f"  [{i}] tensor shape={tuple(o.shape)} dtype={o.dtype} sample={o.flatten()[:5].tolist()}")
    elif isinstance(o, tuple):
        print(f"  [{i}] nested tuple len={len(o)}")
        for j, oo in enumerate(o):
            _describe(oo, f"{i}.{j}")
    else:
        print(f"  [{i}] type={type(o)} val={o}")

if __name__ == "__main__":
    inspect("g1_motion.pt", 47)
    inspect("go2_cts_150k.pt", 45)
    inspect("go2_moe_cts_137k.pt", 45)
