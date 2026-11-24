#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int aclError;
typedef void* aclrtBinHandle;
typedef void* aclrtFuncHandle;
typedef void* aclrtStream;

typedef struct {
    int type;
    uint64_t value;
} aclrtBinaryLoadOption;

typedef struct {
    aclrtBinaryLoadOption* options;
    size_t numOpt;
} aclrtBinaryLoadOptions;

static const char *SFA_O_PATH = "/opt/kernel-overlay/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/sparse_flash_attention/SparseFlashAttention_2e25ad8fefac1f04c03bd78f2d2ab366.o";
static aclrtBinaryLoadOption saved_opts[16];
static aclrtBinaryLoadOptions saved_options = {0};
static int has_saved_options = 0;

static aclrtBinHandle orig_bin = NULL;
static aclrtBinHandle reloaded_bin = NULL;

static aclError (*real_load_data)(const void*, size_t, const aclrtBinaryLoadOptions*, aclrtBinHandle*) = NULL;
static aclError (*real_unload)(aclrtBinHandle) = NULL;
static aclError (*real_get_fn)(const aclrtBinHandle, const char*, aclrtFuncHandle*) = NULL;
static aclError (*real_get_fn_entry)(aclrtBinHandle, uint64_t, aclrtFuncHandle*) = NULL;

aclError aclrtBinaryLoadFromData(const void* data, size_t length, const aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle) {
    if (!real_load_data) real_load_data = dlsym(RTLD_NEXT, "aclrtBinaryLoadFromData");
    aclError ret = real_load_data(data, length, options, binHandle);
    if (ret == 0 && data && length > 64 && memmem(data, length, "SparseFlashAttention_2e25ad8fefac1f04c03bd78f2d2ab366", 53)) {
        orig_bin = *binHandle;
        if (options && options->numOpt <= 16) {
            memcpy(saved_opts, options->options, options->numOpt * sizeof(aclrtBinaryLoadOption));
            saved_options.options = saved_opts;
            saved_options.numOpt = options->numOpt;
            has_saved_options = 1;
        }
    }
    return ret;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle) {
    if (!real_get_fn) real_get_fn = dlsym(RTLD_NEXT, "aclrtBinaryGetFunction");
    aclrtBinHandle target = (binHandle == orig_bin && reloaded_bin != NULL) ? reloaded_bin : binHandle;
    return real_get_fn(target, kernelName, funcHandle);
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle) {
    if (!real_get_fn_entry) real_get_fn_entry = dlsym(RTLD_NEXT, "aclrtBinaryGetFunctionByEntry");
    aclrtBinHandle target = (binHandle == orig_bin && reloaded_bin != NULL) ? reloaded_bin : binHandle;
    return real_get_fn_entry(target, funcEntry, funcHandle);
}

int sfa_reload_binary(void) {
    if (!orig_bin) return -1;
    if (!real_unload) real_unload = dlsym(RTLD_NEXT, "aclrtBinaryUnLoad");
    FILE *fp = fopen(SFA_O_PATH, "rb");
    if (!fp) return -2;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void *buf = malloc(fsize);
    if (!buf || fread(buf, 1, fsize, fp) != (size_t)fsize) {
        fclose(fp);
        free(buf);
        return -3;
    }
    fclose(fp);
    if (reloaded_bin) {
        real_unload(reloaded_bin);
        reloaded_bin = NULL;
    }
    aclrtBinHandle new_bin = NULL;
    aclError ret = real_load_data(buf, (size_t)fsize, has_saved_options ? &saved_options : NULL, &new_bin);
    free(buf);
    if (ret != 0) return (int)ret;
    reloaded_bin = new_bin;
    return 0;
}
