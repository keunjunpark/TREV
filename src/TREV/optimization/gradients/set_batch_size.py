import torch
from contextlib import contextmanager

@contextmanager
def _no_grad_inference():
    # If your batch eval does not require grads, keep this. If it does, remove.
    with torch.no_grad():
        yield

def _gpu_free_bytes(device: torch.device) -> int:
    # mem_get_info is available in recent PyTorch; fallback if needed
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        return int(free_b)
    return 0

def auto_batch_size(
    run_batch_fn,
    device: torch.device,
    *,
    min_bs: int = 1,
    max_bs: int = 65536,
    safety_frac: float = 0.85,
    growth: float = 2.0,
    warmup: int = 2,
    use_amp: bool = False,
) -> int:
    """
    run_batch_fn(bs) should execute *exactly* what a typical TREV batched evaluation does
    (forward/contract/sample/etc.) for that batch size on `device`.
    Returns the largest stable batch size.
    """
    assert min_bs >= 1
       # ---- Normalize device (string -> torch.device with index if CUDA)
    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" or not torch.cuda.is_available():
        return min_bs

    # Ensure CUDA device has explicit index
    if device.index is None:
        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
    else:
        idx = device.index

    torch.cuda.set_device(idx)
    torch.cuda.empty_cache()

    # Try to keep some headroom so fragmentation/caches don’t bite you.
    free_b = _gpu_free_bytes(device)
    target_free_b = int(free_b * safety_frac)

    # Quick warmup at min_bs to initialize kernels/caches
    def _try(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            # Optional: AMP can reduce memory if you can run in fp16/bf16 safely
            if use_amp:
                with _no_grad_inference(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    for _ in range(warmup):
                        run_batch_fn(bs)
            else:
                with _no_grad_inference():
                    for _ in range(warmup):
                        run_batch_fn(bs)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            # If peak allocation already exceeds our target headroom, treat as fail
            return peak < target_free_b
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                # Clear allocator state after OOM
                torch.cuda.empty_cache()
                return False
            raise

    # 1) Exponential growth to find an upper bound
    bs = min_bs
    last_ok = min_bs if _try(min_bs) else 0
    if last_ok == 0:
        return min_bs  # nothing else to do

    while bs < max_bs:
        bs_next = min(int(bs * growth), max_bs)
        if bs_next == bs:
            break
        if _try(bs_next):
            last_ok = bs_next
            bs = bs_next
        else:
            hi = bs_next
            lo = last_ok
            break
    else:
        return last_ok

    # 2) Binary search between last_ok and first_fail
    hi = hi
    lo = last_ok
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid

    return lo
