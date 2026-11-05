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
    __aicore__ inline void Publish(GM_ADDR countsAddress, TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            if (!enabled || GetBlockIdx() != 0) {
                return;
            }
            TBuf<TPosition::VECCALC> buffer;
            pipe->InitBuffer(buffer, (EXPERTS + WORDS) * sizeof(uint64_t));
            auto counts = buffer.Get<int64_t>();
            auto output = buffer.GetWithOffset<uint64_t>(WORDS, EXPERTS * sizeof(uint64_t));
            GlobalTensor<int64_t> input;
            input.SetGlobalBuffer((__gm__ int64_t *)countsAddress);
            DataCopy(counts, input, EXPERTS);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            bool wide = true;
            uint32_t splitExperts = 0;
            uint32_t active = 0;
            uint32_t end = 0;
            for (uint32_t expert = 0; expert < EXPERTS; ++expert) {
                const auto rows = counts.GetValue(expert);
                if (rows > 0) {
                    wide = wide && rows <= 32;
                    splitExperts += rows > 16;
                    end += rows;
                    output.SetValue(++active, (static_cast<uint64_t>(end) << 32) | expert);
                }
            }
            // Split groups need enough parallel work to offset weight rereads.
            // Limit the added passes to one eighth of the active experts.
            wide = wide && (splitExperts == 0 ||
                (end == 256 && active >= 64 && splitExperts * 8 <= active));
            output.SetValue(0, (static_cast<uint64_t>(wide) << 32) | active);
            const uint32_t length = ((active + 4) / 4) * 4;
            for (uint32_t i = active + 1; i < length; ++i) {
                output.SetValue(i, 0);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(records, output, length);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            pipe->Reset();
        }
    }

};
int main() {
    size_t number;
    std::cin >> number;
    constexpr uint64_t poison = 0xdeadbeefcafe1234ULL;
    std::vector<int64_t> counts(256);
    std::vector<uint64_t> metadata(261, poison);
    Publisher publisher;
    publisher.records.SetGlobalBuffer(metadata.data());
    for (size_t step = 0; step < number; ++step) {
        std::string name;
        bool expected;
        std::cin >> name >> expected;
        for (auto &count : counts) std::cin >> count;
        std::fill(metadata.begin(), metadata.end(), poison);
        TPipe pipe;
        publisher.Publish(reinterpret_cast<GM_ADDR>(counts.data()), &pipe);
        uint32_t active = 0, end = 0;
        for (size_t expert = 0; expert < counts.size(); ++expert) {
            if (counts[expert] > 0) {
                end += counts[expert];
                assert(metadata[++active] == (static_cast<uint64_t>(end) << 32 | expert));
            }
        }
        assert(metadata[0] == (static_cast<uint64_t>(expected) << 32 | active));
        assert(static_cast<uint32_t>(metadata[0]) == active);
        const size_t length = ((active + 4) / 4) * 4;
        for (size_t i = active + 1; i < length; ++i) assert(metadata[i] == 0);
        for (size_t i = length; i < metadata.size(); ++i) assert(metadata[i] == poison);
        std::cout << "{\"case\":\"" << name << "\",\"active\":" << active
                  << ",\"end\":" << end << ",\"wide\":" << (metadata[0] >> 32)
                  << ",\"packed_records_and_padding_exact\":true}\n";
    }
}
