import sys; sys.path.insert(0, "/work/runtime-sfa-token-mlatail")
import os, socket, ctypes, torch, torch_npu
import vllm_ascend.vllm_ascend_C

lib = ctypes.CDLL("/tmp/libhotreload_sfa.so")
lib.sfa_reload_binary.argtypes = [ctypes.c_char_p]
lib.sfa_reload_binary.restype = ctypes.c_int

BIN_PATH = b"/opt/kernel-overlay/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_attention/SparseFlashAttention_2e25ad8fefac1f04c03bd78f2d2ab366.o"

torch.npu.set_device(0)
device = torch.device("npu:0")
torch.manual_seed(42)

pages = 73
block_size = 128
kv_len = 8224
topk_len = 2048
scale = 0.0625

key_cache = torch.randn((pages, block_size, 1, 512), dtype=torch.bfloat16, device=device)
key_rope_cache = torch.randn((pages, block_size, 1, 64), dtype=torch.bfloat16, device=device)
block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
act_kv = torch.tensor([kv_len], dtype=torch.int32, device=device)

save_dir = "/tmp/sfa_ref"
cases = {}
for name, m in [("mtp_m1", 1), ("target_m32", 32)]:
    q_c, qr_c, si_c = torch.load(f"{save_dir}/{name}_in.pt")
    q = q_c.to(device).contiguous()
    qr = qr_c.to(device).contiguous()
    sparse_indices = si_c.to(device).contiguous()
    act_q = torch.tensor([m], dtype=torch.int32, device=device)
    ref = torch.load(f"{save_dir}/{name}.pt").to(device)
    cases[name] = (q, qr, sparse_indices, act_q, ref)

torch.npu.synchronize()

def call_sfa(q, qr, sparse_indices, act_q):
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q, key=key_cache, value=key_cache, sparse_indices=sparse_indices,
        scale_value=scale, sparse_block_size=1, block_table=block_table,
        actual_seq_lengths_query=act_q, actual_seq_lengths_kv=act_kv,
        query_rope=qr, key_rope=key_rope_cache,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3,
        attention_mode=2, return_softmax_lse=False
    )
    return out[0] if isinstance(out, (tuple, list)) else out

def run_bench():
    lib.sfa_reload_binary(BIN_PATH)
    res = []
    for name, (q, qr, sparse_indices, act_q, ref) in cases.items():
        out = call_sfa(q, qr, sparse_indices, act_q)
        torch.npu.synchronize()
        max_diff = (out.float() - ref.float()).abs().max().item()
        s = torch.npu.Stream()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, stream=s):
            for _ in range(20):
                call_sfa(q, qr, sparse_indices, act_q)
        for _ in range(3):
            g.replay()
        torch.npu.synchronize()
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        ev0.record()
        for _ in range(10):
            g.replay()
        ev1.record()
        torch.npu.synchronize()
        us = ev0.elapsed_time(ev1) * 1000.0 / 200.0
        res.append(f"{name}: {us:.2f} us (max_diff={max_diff:.2e})")
    return " | ".join(res)

run_bench()

SOCK = "/tmp/sfa_daemon.sock"
if os.path.exists(SOCK):
    os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(4)
with open("/tmp/sfa_daemon.pid", "w") as f:
    f.write(str(os.getpid()))
print("READY", flush=True)
while True:
    conn, _ = srv.accept()
    try:
        out = run_bench()
        conn.sendall((out + "\n").encode())
    except Exception as e:
        conn.sendall((f"ERROR: {e}\n").encode())
    finally:
        conn.close()
