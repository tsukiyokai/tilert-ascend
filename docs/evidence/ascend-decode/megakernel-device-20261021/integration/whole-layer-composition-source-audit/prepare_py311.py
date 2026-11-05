"""Copy a982 and retain an AST-equivalent Python 3.11 quoting change."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys


RELATIVE = Path("vllm/model_executor/model_loader/sharded_state_loader.py")
BEFORE = '''                f"'{
                    os.path.join(
                        local_model_path,
                        rendered_pattern,
                    )
                }', "'''
AFTER = '''                f"""'{
                    os.path.join(
                        local_model_path,
                        rendered_pattern,
                    )
                }', """'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use the existing Python 3.12 interpreter for both ASTs")
    if args.output.exists() or args.report.exists():
        parser.error("Use new output and report paths")
    original_path = args.source / RELATIVE
    original = original_path.read_text()
    if original.count(BEFORE) != 1:
        raise ValueError("The pinned source must contain the exact multiline f-string once")
    changed = original.replace(BEFORE, AFTER, 1)
    old_ast = ast.dump(ast.parse(original), include_attributes=False)
    new_ast = ast.dump(ast.parse(changed), include_attributes=False)
    if old_ast != new_ast or len(original.splitlines()) != len(changed.splitlines()):
        raise ValueError("The complete file AST or line count changed")
    shutil.copytree(args.source, args.output, symlinks=True)
    (args.output / RELATIVE).write_text(changed)
    original_files = {str(p.relative_to(args.source)): digest(p)
                      for p in sorted(args.source.rglob("*")) if p.is_file()}
    changed_files = {str(p.relative_to(args.output)): digest(p)
                     for p in sorted(args.output.rglob("*")) if p.is_file()}
    if original_files.keys() != changed_files.keys():
        raise ValueError("The compatibility copy has a different file inventory")
    differences = [name for name in original_files
                   if original_files[name] != changed_files[name]]
    if differences != [str(RELATIVE)]:
        raise ValueError(f"Unexpected source differences: {differences}")
    report = {
        "python": sys.version,
        "source": str(args.source),
        "output": str(args.output),
        "vllm_commit": "a98247ab4db686ee03c66d5feb3c761e52a2f8ab",
        "changed_files": differences,
        "file_count": len(original_files),
        "original_sha256": original_files[str(RELATIVE)],
        "compatible_sha256": changed_files[str(RELATIVE)],
        "ast_equal_without_source_locations": True,
        "original_ast_sha256": hashlib.sha256(old_ast.encode()).hexdigest(),
        "compatible_ast_sha256": hashlib.sha256(new_ast.encode()).hexdigest(),
        "line_count_unchanged": True,
        "original_file_manifest_sha256": hashlib.sha256(
            json.dumps(original_files, sort_keys=True).encode()).hexdigest(),
        "compatible_file_manifest_sha256": hashlib.sha256(
            json.dumps(changed_files, sort_keys=True).encode()).hexdigest(),
        "script_sha256": digest(Path(__file__)),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.report.with_suffix(".patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile=f"original/{RELATIVE}", tofile=f"compatible/{RELATIVE}")))
    args.report.with_suffix(".files.json").write_text(
        json.dumps({"original": original_files, "compatible": changed_files}, indent=2) + "\n")


if __name__ == "__main__":
    main()
