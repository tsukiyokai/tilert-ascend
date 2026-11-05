import json,pathlib,subprocess
p=pathlib.Path(__file__).parent
r=subprocess.run(json.loads((p/"source-command.json").read_text()),check=False)
raise SystemExit(r.returncode)
