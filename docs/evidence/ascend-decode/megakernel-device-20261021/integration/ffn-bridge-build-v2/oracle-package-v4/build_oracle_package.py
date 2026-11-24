#!/usr/bin/env python3
"""Package two unmodified pinned operators with the installed CANN template."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


OPERATORS = ("moe_gating_top_k", "add_rms_norm_bias")
PINNED_COMMIT = "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_sources(source, destination, identities, prefix):
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise RuntimeError(f"Conflicting source basename: {relative}")
            shutil.copyfile(path, target)
            identities[str(Path(prefix) / relative)] = digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = args.output / "project"
    template = args.cann / "tools/op_project_templates/ascendc/customize"
    report = {"vendor_commit": PINNED_COMMIT, "operators": OPERATORS,
              "script_sha256": digest(Path(__file__)), "official_sources": {},
              "template_sources": {}, "commands": [], "complete": False,
              "device_validation": "not run"}
    copy_sources(template, project, report["template_sources"], "template")
    source = args.vendor / "csrc"
    for operator in OPERATORS:
        directory = source / "moe" / operator
        for kind in ("op_host", "op_kernel"):
            copy_sources(directory / kind, project / "official" / operator / kind,
                         report["official_sources"], operator + "/" + kind)
            for path in (directory / kind).rglob("*"):
                if not path.is_file() or path.name == "CMakeLists.txt":
                    continue
                if kind == "op_host" and path.name.endswith("_proto.cpp"):
                    continue
                relative = path.relative_to(directory / kind)
                target = project / kind / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise RuntimeError(f"Conflicting operator source: {relative}")
                shutil.copyfile(path, target)
    copy_sources(source / "common/include", project / "common/include",
                 report["official_sources"], "common/include")
    log_header = project / "common/include/log/ops_log.h"
    if not log_header.exists():
        log_header.parent.mkdir(parents=True, exist_ok=True)
        log_header.write_text(
            '#pragma once\n#include "tiling_base/error_log.h"\n'
            '#define OPS_LOG_I(context, ...) OP_LOGI(context, __VA_ARGS__)\n'
            '#define OPS_LOG_D(context, ...) OP_LOGD(context, __VA_ARGS__)\n')
        report["compatibility"] = {
            "common/include/log/ops_log.h":
                "CANN 9.2 lacks this include path; forward to the pinned logging "
                "macros already included by both norm tiling translation units"}

    # CANN's template generates the ACLNN ABI and operator metadata. The
    # arithmetic, tilers and schemas above remain the pinned source bytes.
    top = project / "CMakeLists.txt"
    text = top.read_text()
    anchor = "include(cmake/intf.cmake)"
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected CANN project template")
    text = text.replace(anchor, anchor + """
include_directories(
    ${CMAKE_SOURCE_DIR}/common/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common)
add_compile_options(-include${CMAKE_SOURCE_DIR}/common/include/cann_compat.h)
""")
    top.write_text(text)
    host = project / "op_host/CMakeLists.txt"
    host_text = host.read_text()
    host_anchor = "opbuild(OPS_SRC ${ops_srcs}"
    if host_text.count(host_anchor) != 1:
        raise RuntimeError("Unexpected CANN host template")
    host.write_text(host_text.replace(
        host_anchor,
        'file(GLOB op_definitions "${CMAKE_CURRENT_SOURCE_DIR}/*_def.cpp")\n'
        "opbuild(OPS_SRC ${op_definitions}"))
    (project / "op_kernel/CMakeLists.txt").write_text("""
add_ops_compile_options(MoeGatingTopK OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations)
add_ops_compile_options(AddRmsNormBias OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
add_kernels_compile()
""")
    build = args.output / "build"
    install = args.output / "install"
    commands = [
        ["cmake", "-S", str(project), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_SOURCE_PACKAGE=ON",
         "-DENABLE_BINARY_PACKAGE=ON", "-DENABLE_TEST=OFF",
         "-DENABLE_CROSS_COMPILE=OFF", "-DASCEND_PACK_SHARED_LIBRARY=OFF",
         "-DASCEND_COMPUTE_UNIT=ascend910_93",
         "-Dvendor_name=tilert_bridge_oracle",
         f"-DASCEND_CANN_PACKAGE_PATH={args.cann}",
         f"-DCMAKE_INSTALL_PREFIX={install}"],
        ["cmake", "--build", str(build), "--target", "binary", f"-j{args.jobs}"],
        ["cmake", "--build", str(build), f"-j{args.jobs}"],
        ["cmake", "--install", str(build)],
    ]
    manifest = args.output / "manifest.json"
    for index, command in enumerate(commands):
        report["commands"].append(command)
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / f"step-{index}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Official package build failed at step {index}; see its log")
    report["installed_sha256"] = {
        str(path.relative_to(install)): digest(path)
        for path in sorted(install.rglob("*")) if path.is_file()}
    report["project_sha256"] = {
        str(path.relative_to(project)): digest(path)
        for path in sorted(project.rglob("*")) if path.is_file()}
    report["complete"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "install": str(install)}))


if __name__ == "__main__":
    main()
