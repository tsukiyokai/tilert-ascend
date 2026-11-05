import json, pathlib, subprocess
p=pathlib.Path(__file__).parent
c=json.loads((p/"command.json").read_text())
assert "--device" not in c and "--privileged" not in c
with (p/"run.log").open("w") as f:
 r=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=False)
(p/"status.json").write_text(json.dumps({"exit_code":r.returncode})+"\n")
print(json.dumps({"exit_code":r.returncode}),flush=True)
raise SystemExit(r.returncode)
