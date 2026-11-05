// CPU storage model for executing the actual publisher, without pipe simulation.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (true)
using GM_ADDR = unsigned char *;
namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { MTE2_S, S_MTE3, MTE3_S };
constexpr int EVENT_ID0 = 0;
inline unsigned GetBlockIdx() { return 0; }
template <HardEvent> void SetFlag(int) {}
template <HardEvent> void WaitFlag(int) {}
template <typename T> struct LocalTensor {
    T *data;
    size_t size;
    T GetValue(size_t i) { assert(i < size); return data[i]; }
    void SetValue(size_t i, T value) { assert(i < size); data[i] = value; }
};
template <typename T> struct GlobalTensor {
    T *data;
    void SetGlobalBuffer(T *value) { data = value; }
};
template <TPosition> struct TBuf {
    std::vector<uint64_t> storage;
    template <typename T> LocalTensor<T> Get() {
        return {reinterpret_cast<T *>(storage.data()), storage.size() * 8 / sizeof(T)};
    }
    template <typename T> LocalTensor<T> GetWithOffset(size_t n, size_t offset) {
        assert(offset + n * sizeof(T) <= storage.size() * 8);
        return {reinterpret_cast<T *>(reinterpret_cast<char *>(storage.data()) + offset), n};
    }
};
struct TPipe {
    template <TPosition P> void InitBuffer(TBuf<P> &buffer, size_t bytes) {
        buffer.storage.resize((bytes + 7) / 8);
    }
    void Reset() {}
};
template <typename T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, size_t n) {
    assert(n <= dst.size); std::memcpy(dst.data, src.data, n * sizeof(T));
}
template <typename T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t n) {
    assert(n <= src.size); std::memcpy(dst.data, src.data, n * sizeof(T));
}
}
using namespace AscendC;
struct Publisher {
    static constexpr uint32_t EXPERTS = 256, WORDS = 260;
    bool enabled = true;
    GlobalTensor<uint64_t> records;
