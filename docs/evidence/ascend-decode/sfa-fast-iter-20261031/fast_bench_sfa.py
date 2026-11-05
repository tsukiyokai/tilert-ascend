import os, sys, time, socket, subprocess

sock_path = "/tmp/sfa_daemon.sock"
def ensure_daemon():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(2.0)
        s.connect(sock_path)
        return s
    except Exception:
        s.close()
    if os.path.exists(sock_path):
        try: os.unlink(sock_path)
        except Exception: pass
    env = os.environ.copy()
    env["LD_PRELOAD"] = "/tmp/libhotreload_sfa.so"
    env["ASCEND_CUSTOM_OPP_PATH"] = "/opt/kernel-overlay/vendors/custom_transformer"
    subprocess.Popen(
        ["bash", "-lc", "source /opt/ascend/cann-9.2.0/set_env.sh && export ASCEND_CUSTOM_OPP_PATH=/opt/kernel-overlay/vendors/custom_transformer && LD_PRELOAD=/tmp/libhotreload_sfa.so python3 /tmp/sfa_daemon.py"],
        stdin=subprocess.DEVNULL, stdout=open("/tmp/sfa_daemon.log", "w"), stderr=subprocess.STDOUT, start_new_session=True
    )
    for _ in range(60):
        if os.path.exists(sock_path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(10.0)
                s.connect(sock_path)
                return s
            except Exception:
                s.close()
        time.sleep(0.2)
    raise RuntimeError("Failed to start sfa_daemon.py")

s = ensure_daemon()
s.settimeout(10.0)
data = b""
try:
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
except socket.timeout:
    os.system("kill -9 $(pgrep -f sfa_daemon.py) 2>/dev/null || true; rm -f /tmp/sfa_daemon.sock")
    raise TimeoutError("NPU kernel execution timed out in sfa_daemon")
finally:
    s.close()
print(data.decode().strip())
