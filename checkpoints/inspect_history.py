import torch

for path, obs_dim in [("go2_cts_150k.pt", 45), ("go2_moe_cts_137k.pt", 45)]:
    print(f"\n=== {path} ===")
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    print("history attr shape:", m.history.shape, m.history.dtype)
    print("can read via getattr:", torch.is_tensor(getattr(m, "history")))
    # mutate and verify persistence across forward() calls
    with torch.inference_mode():
        h0 = m.history.clone()
        m(torch.randn(1, obs_dim))
        h1 = m.history.clone()
        print("history changed after 1 forward call:", not torch.equal(h0, h1))
        # test manual save/restore round-trip (the session-isolation mechanism)
        saved = m.history.clone()
        m(torch.randn(1, obs_dim))
        h2 = m.history.clone()
        print("history changed after 2nd forward call:", not torch.equal(h1, h2))
        m.history = saved.clone()
        print("restored history matches saved:", torch.equal(m.history, saved))
    # check other mutable non-buffer attrs referenced in code
    for attr in ("history_length", "obs_no_goal_mask"):
        if hasattr(m, attr):
            v = getattr(m, attr)
            print(f"{attr}:", v.shape if torch.is_tensor(v) else v)
