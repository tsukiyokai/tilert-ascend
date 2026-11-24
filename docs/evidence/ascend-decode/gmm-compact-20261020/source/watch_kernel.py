"""Record file-open/access events for the isolated validation kernel."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys

binary, output = map(Path, sys.argv[1:3])
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
libc = ctypes.CDLL(None, use_errno=True)
libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
assert fd >= 0, ctypes.get_errno()
watch = libc.inotify_add_watch(fd, os.fsencode(binary), 0x1 | 0x20)
assert watch >= 0, ctypes.get_errno()
events = []
process = subprocess.Popen(sys.argv[3:])
while True:
    ready, _, _ = select.select([fd], [], [], 0.1)
    if ready:
        data = os.read(fd, 65536)
        offset = 0
        while offset < len(data):
            wd, mask, cookie, size = struct.unpack_from("iIII", data, offset)
            events.append(mask)
            offset += 16 + size
    elif process.poll() is not None:
        break
os.close(fd)
output.write_text(json.dumps({"binary": str(binary), "sha256": digest,
                              "event_masks": events,
                              "opened": any(v & 0x20 for v in events),
                              "accessed": any(v & 0x1 for v in events),
                              "returncode": process.returncode}, indent=2) + "\n")
raise SystemExit(process.returncode)
