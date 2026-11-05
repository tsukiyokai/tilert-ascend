from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
source = root / "original"
target = root / "candidate"
target.mkdir(exist_ok=True)
for path in source.glob("*.h"):
    shutil.copy2(path, target / path.name)
shutil.copy2(root / "active_groups.h", target / "active_groups.h")
prefix = "grouped_matmul_swiglu_quant_v2"


def replace(name, old, new):
    path = target / f"{prefix}_{name}.h"
    text = path.read_text()
    assert text.count(old) == 1, (name, old[:80], text.count(old))
    path.write_text(text.replace(old, new))


replace("utils", "    GM_ADDR workSpaceGM;", "    GM_ADDR activeGroupsGM = nullptr;\n    GM_ADDR workSpaceGM;")
for kind in ("mid", "post"):
    name = f"a8w4_msd_{kind}"
    replace(name, f'#include "{prefix}_utils.h"',
            f'#include "{prefix}_utils.h"\n#include "active_groups.h"')
    replace(name, "    GlobalTensor<int64_t> groupListGM;",
            "    GlobalTensor<int64_t> groupListGM;\n    ActiveGroups activeGroups;")
    replace(name, "        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;",
            "        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;\n"
            "        activeGroups.Init(gmAddrParams.activeGroupsGM);")

name = "a8w4_msd_pipeline"
replace(name, "    InitWorkSpaceSplitConfig(workspaceSplitConfig);",
        """    InitWorkSpaceSplitConfig(workspaceSplitConfig);
    const float limit = gmmSwigluQuantV2BaseParams->swigluLimit;
    uint32_t limitBits = 0;
    __builtin_memcpy(&limitBits, &limit, sizeof(limitBits));
    if (workspaceSplitConfig.loopCount == 1 &&
        gmmSwigluQuantV2BaseParams->groupListType == 1 &&
        gmmSwigluQuantV2->groupListLen == ActiveGroups::EXPERTS &&
        gmmSwigluQuantV2BaseParams->K == 6144 &&
        gmmSwigluQuantV2BaseParams->N == 256 &&
        gmmSwigluQuantV2BaseParams->M <= 256 && limitBits == 0x80000000U) {
        gmAddrParams.activeGroupsGM = workspace + gmAddrParams.workSpaceOffset1;
    }""")
replace(name, "        preProcess.Process(workspaceSplitConfig, 0, pipe);",
        """        ActiveGroups activeGroups;
        activeGroups.Init(gmAddrParams.activeGroupsGM);
        activeGroups.Publish(gmAddrParams.groupListGM, pipe);
        preProcess.Process(workspaceSplitConfig, 0, pipe);""")

name = "a8w4_msd_mid"
replace(name, "        for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex, preCount = 0;\n"
              "             groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; ++groupIdx) {\n"
              "            UpdateMnConfig(mnConfig);\n"
              "            int32_t currSplitValue = 0;\n"
              "            if (gmmSwigluQuantV2BaseParams->groupListType == 0) {",
        """        if (activeGroups.enabled) {
            activeGroups.Invalidate();
        }
        const uint32_t groups = activeGroups.enabled ? activeGroups.Size() :
            workspaceSplitConfig.rightMatrixExpertEndIndex - workspaceSplitConfig.rightMatrixExpertStartIndex + 1;
        for (uint32_t slot = 0, preCount = 0; slot < groups; ++slot) {
            uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex + slot;
            UpdateMnConfig(mnConfig);
            int32_t currSplitValue = 0;
            if (activeGroups.enabled) {
                const uint64_t record = activeGroups.Get(slot);
                groupIdx = static_cast<uint32_t>(record);
                currSplitValue = record >> 32;
            } else if (gmmSwigluQuantV2BaseParams->groupListType == 0) {""")

name = "a8w4_msd_post"
replace(name, "    ActiveGroups activeGroups;", "    ActiveGroups activeGroups;\n    uint32_t activeSlot = 0;")
path = target / f"{prefix}_{name}.h"
text = path.read_text()
start = text.index("    int64_t curStartIdx = vecConfig.startIdx;")
end = text.index("    // 第三步", start)
old = text[start:end]
new = """    if (activeGroups.enabled) {
        activeGroups.Invalidate();
        if (blockIdx < vecConfig.usedCoreNum) {
            activeSlot = activeGroups.FindRow(vecConfig.startIdx);
            const uint64_t record = activeGroups.Get(activeSlot);
            vecConfig.curGroupIdx = static_cast<uint32_t>(record);
            vecConfig.nextUpdateInterVal = (record >> 32) - vecConfig.startIdx;
        }
    } else {
""" + "".join("    " + line if line.strip() else line for line in old.splitlines(keepends=True)) + "    }\n"
text = text[:start] + new + text[end:]
path.write_text(text)
replace(name, "    if (unlikely(vecConfig.nextUpdateInterVal == 0)) {\n"
              "        int64_t loop = gmmSwigluQuantV2->groupListLen - vecConfig.curGroupIdx;",
        """    if (unlikely(vecConfig.nextUpdateInterVal == 0)) {
        if (activeGroups.enabled) {
            const uint32_t previousEnd = activeGroups.End(activeSlot);
            const uint64_t record = activeGroups.Get(++activeSlot);
            vecConfig.curGroupIdx = static_cast<uint32_t>(record);
            vecConfig.nextUpdateInterVal = (record >> 32) - previousEnd;
            PreLoadAuxiliaryMatrix(vecConfig);
            return;
        }
        int64_t loop = gmmSwigluQuantV2->groupListLen - vecConfig.curGroupIdx;""")
print(target)
