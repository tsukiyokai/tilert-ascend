"""Build a tracked-node control with the shared native lookup builder."""

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "lookup-native/build.py"
SPEC = importlib.util.spec_from_file_location("lookup_native_builder", PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
prepare_native = builder.prepare


def prepare(root, output):
    identity, source, paths = prepare_native(root, output)
    identity.update(acyclic_nodes=False,
                    lowering_sha256=builder.digest(Path(__file__)))
    original = paths[0].read_text()
    (source / builder.DIRECTORY / "lookup_draft.py").write_text(
        original + builder.native_guard(identity),
    )
    return identity, source, paths


if __name__ == "__main__":
    builder.prepare = prepare
    raise SystemExit(builder.main())
