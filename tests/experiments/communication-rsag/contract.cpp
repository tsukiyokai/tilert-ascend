#include "control/source/communication.h"
#include "candidate/source/communication.h"
#include "control/source/tree_plan.h"
#include "candidate/source/tree_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsag = trt_rsag_candidate_contract;

void Require(bool valid, const char* message)
{
    if (!valid) throw std::runtime_error(message);
}

template <class Operation> void Reject(Operation operation)
{
    bool rejected = false;
    try { operation(); } catch (const std::runtime_error&) { rejected = true; }
    Require(rejected, "negative schedule was not rejected");
}

uint32_t Bits(float value)
{
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float Add(float left, float right)
{
    volatile float result = left + right;
    return result;
}

float FromBf16(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t ToBf16(float value)
{
    uint32_t bits = Bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return static_cast<uint16_t>((bits >> 16) | 0x40U);
    return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float Tree(const std::vector<float>& values, bool candidate)
{
    std::array<float, 5> stack{};
    std::array<bool, 16> used{};
    int depth = 0;
    for (int leaf = 0; leaf < values.size(); ++leaf) {
        uint32_t rank, reductions;
        if (candidate) {
            const auto step = trt_rsag_candidate::GetTreeStep(
                trt_rsag_candidate::GetTreePlan(values.size()), leaf);
            rank = step.rank; reductions = step.reductions;
        } else {
            const auto step = trt_rsag_control::GetTreeStep(
                trt_rsag_control::GetTreePlan(values.size()), leaf);
            rank = step.rank; reductions = step.reductions;
        }
        Require(rank < values.size() && !used[rank] && depth < 5,
                "invalid tree leaf or stack depth");
        used[rank] = true;
        stack[depth++] = values[rank];  // A source is widened once, no +0 seed.
        for (uint32_t merge = 0; merge < reductions; ++merge) {
            Require(depth >= 2, "tree stack underflow");
            stack[depth - 2] = Add(stack[depth - 2], stack[depth - 1]);
            --depth;
        }
    }
    Require(depth == 1, "incomplete tree");
    return stack[0];
}

float IndependentRounds(std::vector<float> values)
{
    while (values.size() > 1) {
        size_t split = 1;
        while (split * 2 < values.size()) split *= 2;
        for (size_t index = 0; index + split < values.size(); ++index)
            values[index] = Add(values[index], values[index + split]);
        values.resize(split);
    }
    return values.front();
}

int LayoutChecks()
{
    int geometries = 0;
    Require(rsag::AlignedPayload(4096), "aligned base rejected");
    Require(!rsag::AlignedPayload(4098) &&
            !rsag::AlignedPayload(4352) &&
            !rsag::AlignedPayload(0), "shifted/null base accepted");
    for (int capacity : {1, 12, 32}) for (int rows = 1; rows <= capacity; ++rows) {
        for (int ranks = 2; ranks <= 16; ++ranks) {
            const auto layout = trt_rsag_candidate::MakeCommunicationLayout(capacity, ranks);
            const auto control = trt_rsag_control::MakeCommunicationLayout(capacity, ranks);
            Require(layout.scatter % 512 == 0 && layout.gather % 512 == 0 &&
                    layout.scatter_slot_bytes % 512 == 0 &&
                    layout.scatter_plane_bytes % 512 == 0 &&
                    layout.gather_plane_bytes % 512 == 0 &&
                    control.payload % 512 == 0 && control.plane_bytes % 512 == 0,
                    "payload line alignment failed");
            Require(layout.scatter >= layout.gather_ready + 2 * ranks * 64 &&
                    layout.gather >= layout.scatter + 2 * layout.scatter_plane_bytes &&
                    layout.bytes == layout.gather + 2 * layout.gather_plane_bytes,
                    "window regions overlap");
            int64_t cursor = 0;
            for (int owner = 0; owner < ranks; ++owner) {
                const auto range = rsag::OwnerRange(rows, ranks, owner);
                Require(range.begin == cursor && range.count > 0 &&
                        range.count % 256 == 0 && range.begin % 256 == 0 &&
                        range.count * 2 <= layout.scatter_slot_bytes,
                        "owner partition has a gap, overlap or padded operand");
                for (int64_t begin = 0; begin < range.count; begin += 2048) {
                    const int64_t count = std::min<int64_t>(2048, range.count - begin);
                    Require(count > 0 && count % 256 == 0 &&
                            (begin + count) * 2 <= layout.scatter_slot_bytes &&
                            (range.begin + begin + count) * 2 <= layout.gather_plane_bytes,
                            "tail read exceeds valid owner or window");
                }
                cursor += range.count;
            }
            Require(cursor == rows * 6144, "partition lost output elements");
            ++geometries;
        }
    }
    Require(rsag::OwnerRange(1, 16, 0).count == 512 &&
            rsag::OwnerRange(1, 16, 15).count == 256,
            "M1 TP16 became a uniform 384-element split");
    Require(rsag::MakeLayout(0, 16).bytes == 0 &&
            rsag::MakeLayout(33, 16).bytes == 0 &&
            rsag::MakeLayout(32, 17).bytes == 0 &&
            rsag::OwnerRange(1, 16, 16).begin == -1,
            "out-of-range geometry accepted");
    return geometries;
}

struct Actor {
    uint32_t epoch = 0;
    int phase = 0;
    int peer = 0;
    bool pending_put = false;
    int pending_peer = -1;
    uint32_t reading_scatter = 0;
    uint32_t reading_gather = 0;
    uint32_t output = 0;
};

struct Schedule {
    int ranks;
    uint32_t epochs;
    std::vector<Actor> actors;
    uint32_t scatter[16][2][16]{};
    uint32_t scatter_ready[16][2][16]{};
    uint32_t gather[16][2][16]{};
    uint32_t gather_ready[16][2][16]{};

    Schedule(int count, uint32_t rounds) : ranks(count), epochs(rounds), actors(count) {}

    void Issue(Actor& actor, int peer)
    {
        Require(!actor.pending_put, "UB reused before asynchronous put quiet");
        actor.pending_put = true;
        actor.pending_peer = peer;
    }

    void ScatterWrite(int sender, int owner, uint32_t epoch)
    {
        const uint32_t previous = scatter[owner][epoch & 1U][sender];
        Require(!actors[owner].reading_scatter ||
                actors[owner].reading_scatter != previous || previous == epoch,
                "same-parity scatter overwritten while an owner still reads it");
        scatter[owner][epoch & 1U][sender] = epoch;
    }

    void GatherWrite(int owner, int peer, uint32_t epoch)
    {
        const uint32_t previous = gather[peer][epoch & 1U][owner];
        Require(!actors[peer].reading_gather ||
                actors[peer].reading_gather != previous || previous == epoch,
                "same-parity gather overwritten while output still reads it");
        gather[peer][epoch & 1U][owner] = epoch;
    }

    void Notify(int sender, int peer, uint32_t epoch, bool gathered)
    {
        auto& word = gathered ? gather_ready[peer][epoch & 1U][sender]
                              : scatter_ready[peer][epoch & 1U][sender];
        const auto& receiver = actors[peer];
        const bool still_waiting = gathered
            ? (receiver.phase == 7 || receiver.phase == 8)
            : (receiver.phase == 3 || receiver.phase == 4);
        Require(!word || word == epoch || receiver.epoch != word || !still_waiting,
                "same-parity ready word replaced a live epoch");
        word = epoch;
    }

    bool Ready(int rank, bool gathered) const
    {
        const uint32_t epoch = actors[rank].epoch;
        for (int peer = 0; peer < ranks; ++peer) {
            const uint32_t observed = gathered ? gather_ready[rank][epoch & 1U][peer]
                                                : scatter_ready[rank][epoch & 1U][peer];
            if (observed != epoch) return false;
        }
        return true;
    }

    void Read(int rank, bool gathered) const
    {
        const uint32_t epoch = actors[rank].epoch;
        for (int peer = 0; peer < ranks; ++peer)
            Require((gathered ? gather[rank][epoch & 1U][peer]
                              : scatter[rank][epoch & 1U][peer]) == epoch,
                    "ready was observed before the matching payload completed");
    }

    void Consumer(int rank) const
    {
        Require(actors[rank].output == actors[rank].epoch &&
                !actors[rank].reading_gather, "CUBE consumer preceded final output copy");
    }

    bool Step(int rank)
    {
        auto& actor = actors[rank];
        if (actor.epoch == epochs && actor.phase == 0) return false;
        const uint32_t epoch = actor.epoch;
        switch (actor.phase) {
            case 0:
                Consumer(rank);
                Require(epoch < rsag::kMaximumEpoch, "epoch lifetime exceeded");
                ++actor.epoch; actor.peer = 0; actor.phase = 1;
                return true;
            case 1:  // Scatter issues and their quiet completions, before join 2.
                if (actor.pending_put) {
                    ScatterWrite(rank, actor.pending_peer, epoch);
                    actor.pending_put = false;
                    if (++actor.peer == ranks) { actor.peer = 0; actor.phase = 2; }
                } else Issue(actor, (rank + actor.peer) % ranks);
                return true;
            case 2:  // AIV0's ordered scatter-ready wave after all quiets.
                Notify(rank, (rank + actor.peer) % ranks, epoch, false);
                if (++actor.peer == ranks) { actor.peer = 0; actor.phase = 3; }
                return true;
            case 3:
                if (!Ready(rank, false)) return false;
                Read(rank, false); actor.reading_scatter = epoch; actor.phase = 4;
                return true;
            case 4:  // Split start/finish exposes overlapping remote writes.
                Read(rank, false); actor.reading_scatter = 0;
                GatherWrite(rank, rank, epoch);
                actor.peer = 1; actor.phase = 5;  // Join 3 before broadcast.
                return true;
            case 5:
                if (actor.pending_put) {
                    GatherWrite(rank, actor.pending_peer, epoch);
                    actor.pending_put = false;
                    if (++actor.peer == ranks) { actor.peer = 0; actor.phase = 6; }
                } else Issue(actor, (rank + actor.peer) % ranks);
                return true;
            case 6:  // Join 4 has drained every broadcast put.
                Notify(rank, (rank + actor.peer) % ranks, epoch, true);
                if (++actor.peer == ranks) { actor.peer = 0; actor.phase = 7; }
                return true;
            case 7:
                if (!Ready(rank, true)) return false;
                Read(rank, true); actor.reading_gather = epoch; actor.phase = 8;
                return true;
            case 8:
                Read(rank, true); actor.reading_gather = 0;
                actor.output = epoch; actor.phase = 0;  // Join 5 then consumer.
                Consumer(rank);
                return true;
        }
        throw std::runtime_error("unknown phase");
    }
};

void ScheduleChecks()
{
    std::mt19937 random(20261028);
    for (int ranks = 2; ranks <= 16; ++ranks) for (int run = 0; run < 40; ++run) {
        Schedule schedule(ranks, 8);
        int idle = 0;
        while (!std::all_of(schedule.actors.begin(), schedule.actors.end(),
                           [](const Actor& actor) { return actor.epoch == 8 && actor.phase == 0; })) {
            const int rank = random() % ranks;
            if (schedule.Step(rank)) idle = 0;
            else Require(++idle < 10000, "protocol stalled with consistent epochs");
        }
    }
    Schedule broken(2, 8);
    broken.actors[0].epoch = 1;
    broken.scatter_ready[0][1][0] = broken.scatter_ready[0][1][1] = 1;
    Reject([&] { broken.Read(0, false); });  // Ready before scatter completion.
    broken.Issue(broken.actors[0], 1);
    Reject([&] { broken.Issue(broken.actors[0], 0); });  // Missing quiet.
    broken.scatter[1][1][0] = 1; broken.actors[1].reading_scatter = 1;
    Reject([&] { broken.ScatterWrite(0, 1, 3); });
    broken.gather[1][1][0] = 1; broken.actors[1].reading_gather = 1;
    Reject([&] { broken.GatherWrite(0, 1, 3); });
    Reject([&] { broken.Consumer(1); });
    broken.actors[1].epoch = 1; broken.actors[1].phase = 3;
    broken.scatter_ready[1][1][0] = 1;
    Reject([&] { broken.Notify(0, 1, 3, false); });
    broken.actors[1].phase = 7; broken.gather_ready[1][1][0] = 1;
    Reject([&] { broken.Notify(0, 1, 3, true); });
    broken.actors[0].pending_put = false;
    broken.actors[0].phase = 0; broken.actors[0].epoch = rsag::kMaximumEpoch;
    broken.actors[0].output = broken.actors[0].epoch;
    Reject([&] { broken.Step(0); });
    // One rank skipping the first epoch cannot satisfy matching EQ flags.
    Schedule inconsistent(2, 8);
    inconsistent.actors[0].epoch = 1;
    inconsistent.actors[1].epoch = 2;
    inconsistent.scatter_ready[0][1][0] = 1;
    inconsistent.scatter_ready[0][1][1] = 2;
    Require(!inconsistent.Ready(0, false), "inconsistent epochs falsely became ready");
}

void ArithmeticChecks()
{
    std::mt19937 random(20261028);
    for (int ranks = 2; ranks <= 16; ++ranks) for (int run = 0; run < 256; ++run) {
        std::vector<float> values;
        for (int rank = 0; rank < ranks; ++rank)
            values.push_back(FromBf16(static_cast<uint16_t>(random())));
        const float control = Tree(values, false), candidate = Tree(values, true);
        Require(Bits(control) == Bits(candidate) &&
                Bits(control) == Bits(IndependentRounds(values)),
                "tree rank multiplicity or FP32 association changed");
    }
    const std::vector<float> zeros{FromBf16(0x8000), FromBf16(0x8000)};
    Require(ToBf16(Tree(zeros, true)) == 0x8000 &&
            Bits(Add(0.0F, zeros[0])) != Bits(zeros[0]), "zero-seed negative failed");
    const std::vector<float> rounding{256.0F, -256.0F, 1.0F, 0.0F};
    Require(Tree(rounding, true) == 1.0F &&
            Add(FromBf16(ToBf16(Add(256.0F, 1.0F))), -256.0F) == 0.0F,
            "intermediate BF16 rounding negative failed");
    const float maximum = FromBf16(0x7f7f);
    Require(std::isnan(Tree({maximum, -maximum, maximum, -maximum}, true)),
            "four-rank FP32 overflow witness disappeared");
}

int main()
{
    try {
        const int geometries = LayoutChecks();
        ArithmeticChecks();
        ScheduleChecks();
        std::cout << "layout_geometries=" << geometries
                  << " arithmetic_cases=3840 schedules=600 epochs_per_rank=8"
                  << " negative_cases=11 CPU-only\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
