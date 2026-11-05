import json
from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
with (p / "run.log").open("w") as output:
    result=subprocess.run(json.loads((p / "command.json").read_text()),stdout=output,stderr=subprocess.STDOUT)
print((p / "run.log").read_text())
raise SystemExit(result.returncode)
