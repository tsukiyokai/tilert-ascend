import json, subprocess
from pathlib import Path
p=Path(__file__).parent
with (p/"run.log").open("w") as stream:
    result=subprocess.run(json.loads((p/"command.json").read_text()),stdout=stream,stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
