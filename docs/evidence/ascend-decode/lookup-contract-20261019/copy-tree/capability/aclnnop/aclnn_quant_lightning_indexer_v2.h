
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_QUANT_LIGHTNING_INDEXER_V2_H_
#define ACLNN_QUANT_LIGHTNING_INDEXER_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnQuantLightningIndexerV2GetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * w : required
 * qDescale : required
 * kDescale : required
 * cuSeqlensQOptional : optional
 * cuSeqlensKOptional : optional
 * sequsedQOptional : optional
 * sequsedKOptional : optional
 * cmpResidualKOptional : optional
 * blockTableOptional : optional
 * outputIdxOffsetOptional : optional
 * metadataOptional : optional
 * topk : required
 * quantMode : required
 * maxSeqlenQ : optional
 * layoutQOptional : optional
 * layoutKOptional : optional
 * maskMode : optional
 * cmpRatio : optional
 * returnValue : optional
 * sparseIndicesOut : required
 * sparseValuesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *qDescale,
    const aclTensor *kDescale,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    int64_t topk,
    int64_t quantMode,
    int64_t maxSeqlenQ,
    char *layoutQOptional,
    char *layoutKOptional,
    int64_t maskMode,
    int64_t cmpRatio,
    int64_t returnValue,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnQuantLightningIndexerV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnQuantLightningIndexerV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
