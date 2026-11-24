// Native, single-host HCCL calibration. Run only on the reserved devices.
#include <acl/acl.h>
#include <hccl/hccl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <poll.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {

constexpr int kRanks = 16;
constexpr int kChainLength = 156;
constexpr uint32_t kCommConfigAiv = 3;  // hccl_types.h: hcclOpExpansionMode.
volatile sig_atomic_t interrupted = 0;

struct Options {
    fs::path output;
    std::vector<int> devices;
    int batches = 100;
    int warmup = 20;
    int iterations = 100;
    int timeout_seconds = 300;
    bool aiv = true;
};

struct Case {
    size_t bytes;
    int operations;
    const char* name;
};

constexpr std::array<Case, 4> cases = {{{12288, 1, "independent_12kib"},
                                      {12288, kChainLength, "chain_12kib"},
                                      {49152, 1, "independent_48kib"},
                                      {49152, kChainLength, "chain_48kib"}}};

enum class Kind : uint32_t { ready = 1, release, batch, done };

struct Message {
    Kind kind = Kind::ready;
    uint32_t case_id = 0;
    int32_t batch = 0;
    int32_t iteration = 0;
    double device_us = 0;
    double host_wall_us = 0;
    double host_release_us = 0;
    double host_enqueue_us = 0;
};

struct Sample {
    double device_us;
    double host_wall_us;
    double host_release_us;
    double host_enqueue_us;
    uint64_t release_ns;
};

struct Summary {
    std::vector<double> device;
    std::vector<double> wall;
    std::vector<double> release;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void acl_check(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        fail(std::string(operation) + " failed: ACL " + std::to_string(status));
    }
}

void hccl_check(HcclResult status, const char* operation) {
    if (status != HCCL_SUCCESS) {
        fail(std::string(operation) + " failed: HCCL " + std::to_string(status));
    }
}

#define ACL_CHECK(call) acl_check((call), #call)
#define HCCL_CHECK(call) hccl_check((call), #call)

uint64_t monotonic_ns() {
    timespec time{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &time) != 0) fail("clock_gettime failed");
    return static_cast<uint64_t>(time.tv_sec) * 1000000000ULL + time.tv_nsec;
}

double elapsed_us(uint64_t start, uint64_t end) {
    return static_cast<double>(end - start) / 1000.0;
}

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 32) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    out << '"';
    return out.str();
}

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << content << '\n';
    out.close();
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) return "";
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int positive_int(const std::string& text, bool allow_zero = false) {
    size_t consumed = 0;
    const long value = std::stol(text, &consumed);
    if (consumed != text.size() || value < (allow_zero ? 0 : 1) || value > 1000000) {
        fail("invalid integer: " + text);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int rank = 0; rank < kRanks; ++rank) options.devices.push_back(rank);
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "Usage: hccl_latency --output NEW_DIRECTORY [--devices 0,1,...,15]\n"
                      << "  --batches 100 --warmup 20 --iterations 100 --timeout-seconds 300\n"
                      << "  --expansion aiv|default (default aiv; communicator config 3)\n"
                      << "Four cases: BF16 SUM, 12/48 KiB, one op/156 in-place dependent ops.\n"
                      << "This executable starts 16 NPU workers. --help initializes no runtime.\n";
            std::exit(0);
        }
        if (i + 1 == argc) fail("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--output") options.output = fs::absolute(value);
        else if (key == "--batches") options.batches = positive_int(value);
        else if (key == "--warmup") options.warmup = positive_int(value, true);
        else if (key == "--iterations") options.iterations = positive_int(value);
        else if (key == "--timeout-seconds") options.timeout_seconds = positive_int(value);
        else if (key == "--expansion") {
            if (value != "aiv" && value != "default") fail("invalid expansion mode");
            options.aiv = value == "aiv";
        } else if (key == "--devices") {
            options.devices.clear();
            std::istringstream input(value);
            std::string item;
            while (std::getline(input, item, ',')) {
                options.devices.push_back(positive_int(item, true));
            }
        } else fail("unknown option: " + key);
    }
    if (options.output.empty()) fail("--output is required");
    auto sorted = options.devices;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() != kRanks || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        fail("--devices must contain exactly 16 distinct logical device IDs");
    }
    if (options.timeout_seconds > 3600) fail("timeout must not exceed 3600 seconds");
    const char* expansion = std::getenv("HCCL_OP_EXPANSION_MODE");
    if (options.aiv && expansion && std::string(expansion) != "AIV") {
        fail("--expansion aiv conflicts with HCCL_OP_EXPANSION_MODE; unset it or use AIV");
    }
    return options;
}

void transfer(int fd, void* data, size_t size, bool sending, int timeout_seconds) {
    auto* current = static_cast<char*>(data);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (size != 0) {
        if (interrupted) fail("interrupted");
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) fail("control socket timed out");
        pollfd item{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
        const int result = poll(&item, 1, static_cast<int>(remaining));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) fail("control socket poll failed or timed out");
        const ssize_t count = sending ? send(fd, current, size, MSG_NOSIGNAL)
                                      : recv(fd, current, size, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) fail("worker/control connection closed; inspect rank-*.error.json");
        current += count;
        size -= static_cast<size_t>(count);
    }
}

void send_message(int fd, Message message, const Options& options) {
    transfer(fd, &message, sizeof(message), true, options.timeout_seconds);
}

Message receive_message(int fd, const Options& options) {
    Message message;
    transfer(fd, &message, sizeof(message), false, options.timeout_seconds);
    return message;
}

std::vector<Message> gather(const std::vector<int>& sockets, const Message& expected,
                            const Options& options) {
    std::vector<Message> messages(kRanks);
    std::vector<pollfd> pending;
    for (int fd : sockets) pending.push_back({fd, POLLIN, 0});
    int received = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_seconds);
    while (received < kRanks) {
        if (interrupted) fail("interrupted");
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) fail("rank rendezvous timed out");
        const int count = poll(pending.data(), pending.size(), static_cast<int>(remaining));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) fail("rank rendezvous poll failed or timed out");
        for (int rank = 0; rank < kRanks; ++rank) {
            if (pending[rank].fd < 0 || pending[rank].revents == 0) continue;
            const auto message = receive_message(sockets[rank], options);
            if (message.kind != expected.kind || message.case_id != expected.case_id ||
                message.batch != expected.batch || message.iteration != expected.iteration) {
                fail("unexpected control message from rank " + std::to_string(rank));
            }
            messages[rank] = message;
            pending[rank].fd = -1;
            ++received;
        }
    }
    return messages;
}

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0xffffU) != 0) fail("payload is not exactly BF16 representable");
    return static_cast<uint16_t>(bits >> 16);
}

float fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint16_t> payload(size_t elements, int rank, bool balanced) {
    std::vector<uint16_t> values(elements);
    const int magnitude = balanced ? (rank < kRanks / 2 ? 1 : -1) : rank + 1;
    for (size_t i = 0; i < elements; ++i) {
        values[i] = bf16(static_cast<float>((i % 2 == 0 ? 1 : -1) * magnitude));
    }
    return values;
}

struct Runtime {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    aclrtStream release_stream = nullptr;
    aclrtEvent gate = nullptr;
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    void* chain = nullptr;
    void* align_input = nullptr;
    void* align_output = nullptr;
};

void all_reduce(Runtime& runtime, void* input, void* output, size_t bytes) {
    HCCL_CHECK(HcclAllReduce(input, output, bytes / sizeof(uint16_t), HCCL_DATA_TYPE_BFP16,
                            HCCL_REDUCE_SUM, runtime.comm, runtime.stream));
}

void synchronize(Runtime& runtime, const Options& options) {
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(runtime.stream, options.timeout_seconds * 1000));
}

void upload(void* destination, const std::vector<uint16_t>& values) {
    const size_t bytes = values.size() * sizeof(uint16_t);
    ACL_CHECK(aclrtMemcpy(destination, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

void validate(void* buffer, size_t bytes, bool zero) {
    std::vector<uint16_t> actual(bytes / sizeof(uint16_t));
    ACL_CHECK(aclrtMemcpy(actual.data(), bytes, buffer, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float expected = zero ? 0.0F : (i % 2 == 0 ? 136.0F : -136.0F);
        if (fp32(actual[i]) != expected) {
            fail("BF16 SUM mismatch at element " + std::to_string(i) + ": got " +
                 std::to_string(fp32(actual[i])) + ", expected " + std::to_string(expected));
        }
    }
}

void preflight(Runtime& runtime, const Case& test_case, int rank, const Options& options) {
    const auto nonzero = payload(test_case.bytes / 2, rank, false);
    upload(runtime.input, nonzero);
    upload(runtime.align_input, nonzero);
    all_reduce(runtime, runtime.input, runtime.output, test_case.bytes);
    synchronize(runtime, options);
    validate(runtime.output, test_case.bytes, false);
    upload(runtime.chain, nonzero);
    all_reduce(runtime, runtime.chain, runtime.chain, test_case.bytes);
    synchronize(runtime, options);
    validate(runtime.chain, test_case.bytes, false);
    upload(runtime.chain, payload(test_case.bytes / 2, rank, true));
    all_reduce(runtime, runtime.chain, runtime.chain, test_case.bytes);
    synchronize(runtime, options);
    validate(runtime.chain, test_case.bytes, true);
}

Sample run_sample(Runtime& runtime, const Case& test_case, const Message& tag, int socket,
                  const std::vector<uint16_t>& balanced, const Options& options) {
    if (test_case.operations != 1) upload(runtime.chain, balanced);
    const uint64_t host_start = monotonic_ns();
    // Same wait/reset/release protocol as CANN 9.2 hccl_test -t 1.
    ACL_CHECK(aclrtStreamWaitEvent(runtime.stream, runtime.gate));
    ACL_CHECK(aclrtResetEvent(runtime.gate, runtime.stream));
    // Complete a same-shape collective before timing to absorb host release skew.
    all_reduce(runtime, runtime.align_input, runtime.align_output, test_case.bytes);
    ACL_CHECK(aclrtRecordEvent(runtime.start, runtime.stream));
    for (int operation = 0; operation < test_case.operations; ++operation) {
        if (test_case.operations == 1) {
            all_reduce(runtime, runtime.input, runtime.output, test_case.bytes);
        } else {
            // Each operation consumes the preceding operation's output in place.
            all_reduce(runtime, runtime.chain, runtime.chain, test_case.bytes);
        }
    }
    ACL_CHECK(aclrtRecordEvent(runtime.end, runtime.stream));
    const uint64_t enqueue_end = monotonic_ns();
    send_message(socket, tag, options);
    const auto release = receive_message(socket, options);
    if (release.kind != Kind::release || release.case_id != tag.case_id ||
        release.batch != tag.batch || release.iteration != tag.iteration) {
        fail("invalid gate release message");
    }
    const uint64_t release_start = monotonic_ns();
    ACL_CHECK(aclrtRecordEvent(runtime.gate, runtime.release_stream));
    synchronize(runtime, options);
    const uint64_t host_end = monotonic_ns();
    float milliseconds = 0;
    ACL_CHECK(aclrtEventElapsedTime(&milliseconds, runtime.start, runtime.end));
    if (!std::isfinite(milliseconds) || milliseconds <= 0) fail("invalid device event duration");
    return {static_cast<double>(milliseconds) * 1000.0, elapsed_us(host_start, host_end),
            elapsed_us(release_start, host_end), elapsed_us(host_start, enqueue_end), release_start};
}

std::string library_path(void* symbol) {
    Dl_info info{};
    return dladdr(symbol, &info) && info.dli_fname ? info.dli_fname : "unknown";
}

void write_rank_metadata(int rank, const Options& options) {
    std::ostringstream out;
    const char* soc = aclrtGetSocName();
    out << "{\n  \"rank\": " << rank << ", \"pid\": " << getpid()
        << ", \"logical_device\": " << options.devices[rank]
        << ", \"soc\": " << quote(soc ? soc : "unknown") << ",\n  \"cpu_affinity\": [";
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    const int affinity_status = sched_getaffinity(0, sizeof(affinity), &affinity);
    bool first = true;
    if (affinity_status == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &affinity)) continue;
            if (!first) out << ',';
            first = false;
            out << cpu;
        }
    }
    out << "], \"affinity_status\": " << affinity_status << ",\n  \"device_attributes\": {";
    const std::array<std::pair<const char*, aclrtDevAttr>, 5> attributes = {{
        {"physical_chip_id", ACL_DEV_ATTR_PHY_CHIP_ID}, {"npu_arch", ACL_DEV_ATTR_NPU_ARCH},
        {"total_memory_bytes", ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE},
        {"super_pod_server_id", ACL_DEV_ATTR_SUPER_POD_SERVER_ID},
        {"super_pod_device_id", ACL_DEV_ATTR_SUPER_POD_DEVICE_ID}}};
    for (size_t i = 0; i < attributes.size(); ++i) {
        int64_t value = -1;
        const auto status = aclrtGetDeviceInfo(options.devices[rank], attributes[i].second, &value);
        if (i) out << ',';
        out << '\n' << quote(attributes[i].first) << ": {\"status\": " << status
            << ", \"value\": " << value << '}';
    }
    out << "\n  }, \"loaded_libraries\": {\"HcclAllReduce\": "
        << quote(library_path(reinterpret_cast<void*>(&HcclAllReduce)))
        << ", \"HcclCommInitRootInfoConfig\": "
        << quote(library_path(reinterpret_cast<void*>(&HcclCommInitRootInfoConfig)))
        << ", \"aclrtRecordEvent\": "
        << quote(library_path(reinterpret_cast<void*>(&aclrtRecordEvent))) << "}\n}";
    write_file(options.output / ("rank-" + std::to_string(rank) + ".json"), out.str());
}

Runtime initialize_worker(int rank, int socket, const Options& options) {
    // The parent never calls ACL/HCCL before fork.
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.devices[rank]));
    HcclRootInfo root{};
    if (rank == 0) {
        HCCL_CHECK(HcclGetRootInfo(&root));
        transfer(socket, &root, sizeof(root), true, options.timeout_seconds);
    }
    transfer(socket, &root, sizeof(root), false, options.timeout_seconds);
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = rank;
    config.hcclBufferSize = 200;
    config.hcclOpExpansionMode = options.aiv ? kCommConfigAiv : HCCL_COMM_DEFAULT_OP_EXPANSION_MODE;
    Runtime runtime;
    HCCL_CHECK(HcclCommInitRootInfoConfig(kRanks, &root, rank, &config, &runtime.comm));
    ACL_CHECK(aclrtCreateStream(&runtime.stream));
    ACL_CHECK(aclrtCreateStream(&runtime.release_stream));
    ACL_CHECK(aclrtSetStreamFailureMode(runtime.stream, 1));
    ACL_CHECK(aclrtCreateEventWithFlag(&runtime.gate, ACL_EVENT_SYNC));
    ACL_CHECK(aclrtCreateEvent(&runtime.start));
    ACL_CHECK(aclrtCreateEvent(&runtime.end));
    for (void** pointer : {&runtime.input, &runtime.output, &runtime.chain,
                           &runtime.align_input, &runtime.align_output}) {
        ACL_CHECK(aclrtMalloc(pointer, cases.back().bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    write_rank_metadata(rank, options);
    return runtime;
}

void cleanup(Runtime& runtime, int device) {
    HCCL_CHECK(HcclCommDestroy(runtime.comm));
    for (void* pointer : {runtime.input, runtime.output, runtime.chain,
                          runtime.align_input, runtime.align_output}) ACL_CHECK(aclrtFree(pointer));
    ACL_CHECK(aclrtDestroyStream(runtime.stream));
    ACL_CHECK(aclrtDestroyStream(runtime.release_stream));
    ACL_CHECK(aclrtDestroyEvent(runtime.gate));
    ACL_CHECK(aclrtDestroyEvent(runtime.start));
    ACL_CHECK(aclrtDestroyEvent(runtime.end));
    ACL_CHECK(aclrtResetDevice(device));
    ACL_CHECK(aclFinalize());
}

void worker(int rank, int socket, const Options& options) {
    auto runtime = initialize_worker(rank, socket, options);
    std::ofstream raw(options.output / ("rank-" + std::to_string(rank) + "-samples.csv"));
    raw.exceptions(std::ios::badbit | std::ios::failbit);
    raw << "case,batch,sample,bytes,operations,device_unit_us,host_wall_unit_us,"
           "host_release_unit_us,host_enqueue_unit_us,release_monotonic_raw_ns\n";
    for (uint32_t case_id = 0; case_id < cases.size(); ++case_id) {
        const auto& test_case = cases[case_id];
        preflight(runtime, test_case, rank, options);
        const auto balanced = payload(test_case.bytes / 2, rank, true);
        for (int batch = 0; batch < options.batches; ++batch) {
            Message means{Kind::batch, case_id, batch, options.iterations};
            std::ostringstream rows;
            rows << std::setprecision(12);
            for (int i = -options.warmup; i < options.iterations; ++i) {
                const Message tag{Kind::ready, case_id, batch, i};
                const auto sample = run_sample(runtime, test_case, tag, socket, balanced, options);
                if (i < 0) continue;
                means.device_us += sample.device_us;
                means.host_wall_us += sample.host_wall_us;
                means.host_release_us += sample.host_release_us;
                means.host_enqueue_us += sample.host_enqueue_us;
                rows << test_case.name << ',' << batch << ',' << i << ',' << test_case.bytes << ','
                     << test_case.operations << ',' << sample.device_us << ',' << sample.host_wall_us
                     << ',' << sample.host_release_us << ',' << sample.host_enqueue_us << ','
                     << sample.release_ns << '\n';
            }
            validate(runtime.align_output, test_case.bytes, false);
            validate(test_case.operations == 1 ? runtime.output : runtime.chain,
                     test_case.bytes, test_case.operations != 1);
            raw << rows.str();
            raw.flush();
            means.device_us /= options.iterations;
            means.host_wall_us /= options.iterations;
            means.host_release_us /= options.iterations;
            means.host_enqueue_us /= options.iterations;
            send_message(socket, means, options);
        }
    }
    raw.close();
    cleanup(runtime, options.devices[rank]);
    send_message(socket, Message{Kind::done}, options);
}

std::string environment_json() {
    constexpr std::array<const char*, 18> names = {{
        "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE", "HCCL_ALGO", "HCCL_DETERMINISTIC",
        "HCCL_EXEC_TIMEOUT", "HCCL_CONNECT_TIMEOUT", "HCCL_LOG_LEVEL", "HCCL_DIAGNOSE_ENABLE",
        "HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE", "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID", "ASCEND_LAUNCH_BLOCKING", "ASCEND_GLOBAL_LOG_LEVEL",
        "ASCEND_SLOG_PRINT_TO_STDOUT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH"}};
    std::ostringstream out;
    out << '{';
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) out << ',';
        const char* value = std::getenv(names[i]);
        out << '\n' << quote(names[i]) << ':' << (value ? quote(value) : "null");
    }
    return out.str() + "\n}";
}

void write_run_metadata(const Options& options, int argc, char** argv) {
    utsname host{};
    if (uname(&host) != 0) fail("uname failed");
    std::array<char, 4096> executable{};
    const ssize_t length = readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length < 0) fail("cannot resolve executable path");
    const fs::path binary(std::string(executable.data(), static_cast<size_t>(length)));
    const auto manifest = read_file(binary.string() + ".build.json");
    const std::time_t started = std::time(nullptr);
    std::ostringstream started_utc;
    started_utc << std::put_time(std::gmtime(&started), "%Y-%m-%dT%H:%M:%SZ");
    std::ostringstream out;
    out << "{\n\"schema\": 1, \"binary\": " << quote(binary.string())
        << ", \"started_utc\": " << quote(started_utc.str())
        << ", \"host\": " << quote(host.nodename) << ", \"kernel\": " << quote(host.release)
        << ", \"machine\": " << quote(host.machine) << ",\n\"argv\": [";
    for (int i = 0; i < argc; ++i) { if (i) out << ','; out << quote(argv[i]); }
    out << "],\n\"ranks\": 16, \"logical_devices\": [";
    for (size_t i = 0; i < options.devices.size(); ++i) { if (i) out << ','; out << options.devices[i]; }
    out << "], \"batches\": " << options.batches << ", \"warmup_units_per_batch\": " << options.warmup
        << ", \"measured_units_per_batch\": " << options.iterations
        << ", \"chain_operations_per_unit\": 156, \"dtype\": \"BF16\", \"reduction\": \"SUM\",\n"
        << "\"communicator\": {\"api\": \"HcclCommInitRootInfoConfig\", \"buffer_mib\": 200, "
        << "\"requested_expansion\": " << quote(options.aiv ? "aiv" : "library_default")
        << ", \"hcclOpExpansionMode\": " << (options.aiv ? kCommConfigAiv : 0)
        << ", \"observed_executor\": null},\n"
        << "\"timing\": {\"gate\": \"ACL_EVENT_SYNC; comm stream wait then reset; release stream record\", "
        << "\"alignment\": \"one untimed same-shape BF16 SUM on separate buffers after gate\", "
        << "\"device\": \"aclrtCreateEvent; start/end bracket only target op(s); aclrtEventElapsedTime\", "
        << "\"host_wall\": \"CLOCK_MONOTONIC_RAW, before gate enqueue through stream completion; includes enqueue, IPC barrier, alignment, gate and sync\", "
        << "\"host_release\": \"before gate record through stream completion; includes alignment and sync\", "
        << "\"host_enqueue\": \"before gate wait enqueue through end-event enqueue\", "
        << "\"release_skew\": \"per-sample host release timestamps recorded; alignment reduces host skew but does not prove simultaneous device starts\", "
        << "\"device_overhead_subtracted\": false, \"graph_replay\": false},\n"
        << "\"aggregation\": \"mean 100 measured units per rank/batch (or CLI count), MAX across 16 rank means, then nearest-rank p50/p95 across batches; chain unit is all 156 ops\",\n"
        << "\"payload\": {\"independent\": \"alternating +/- (rank+1), expected +/-136; separate input/output\", "
        << "\"chain\": \"same input/output pointer for every op; balanced +/-1 by rank, zero after first SUM; restored outside every unit\", "
        << "\"validation\": \"nonzero out-of-place and in-place SUM plus first balanced SUM before timing; alignment and final output checked after each batch\", "
        << "\"scaling_kernel\": false},\n"
        << "\"gate_source\": \"CANN 9.2 toolkit/tools/hccl_test/common/src/hccl_test_common.cc:875-881,1258-1275\",\n"
        << "\"expansion_source\": \"CANN 9.2 include/hccl/hccl_types.h:149; hccl_test_common.cc:890-903\",\n"
        << "\"environment\": " << environment_json() << ",\n\"build\": "
        << (manifest.empty() ? "null" : manifest) << "\n}";
    write_file(options.output / "run.json", out.str());
}

double quantile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(probability * values.size())) - 1;
    return values.at(index);
}

std::string statistics(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::setprecision(12) << "{\"min\":" << *std::min_element(values.begin(), values.end())
        << ",\"p50\":" << quantile(values, 0.50) << ",\"p95\":" << quantile(values, 0.95)
        << ",\"max\":" << *std::max_element(values.begin(), values.end()) << '}';
    return out.str();
}

void write_summary(const Options& options, const std::array<Summary, 4>& summaries) {
    std::ostringstream out;
    out << "{\"status\":\"complete\",\"validation\":\"all ranks passed\","
        << "\"statistic\":\"quantiles of rank-MAX batch means, not individual-op tail latency\","
        << "\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) out << ',';
        auto per_operation = summaries[i].device;
        for (double& value : per_operation) value /= cases[i].operations;
        out << "\n{\"name\":" << quote(cases[i].name) << ",\"bytes\":" << cases[i].bytes
            << ",\"operations_per_unit\":" << cases[i].operations
            << ",\"device_unit_us\":" << statistics(summaries[i].device)
            << ",\"device_amortized_per_op_us\":" << statistics(per_operation)
            << ",\"host_wall_unit_us\":" << statistics(summaries[i].wall)
            << ",\"host_release_unit_us\":" << statistics(summaries[i].release) << '}';
    }
    out << "\n]}";
    write_file(options.output / "summary.json", out.str());
}

std::array<Summary, 4> coordinate(const std::vector<int>& sockets, const Options& options) {
    HcclRootInfo root{};
    transfer(sockets[0], &root, sizeof(root), false, options.timeout_seconds);
    for (int socket : sockets) transfer(socket, &root, sizeof(root), true, options.timeout_seconds);
    std::ofstream ranks(options.output / "rank-batch-means.csv");
    std::ofstream maxima(options.output / "rank-max-batch-means.csv");
    ranks.exceptions(std::ios::badbit | std::ios::failbit);
    maxima.exceptions(std::ios::badbit | std::ios::failbit);
    ranks << "case,batch,rank,device_unit_us,host_wall_unit_us,host_release_unit_us,host_enqueue_unit_us\n";
    maxima << "case,batch,device_unit_us,device_max_rank,host_wall_unit_us,wall_max_rank,host_release_unit_us\n";
    ranks << std::setprecision(12);
    maxima << std::setprecision(12);
    std::array<Summary, 4> summaries;
    for (uint32_t case_id = 0; case_id < cases.size(); ++case_id) {
        for (int batch = 0; batch < options.batches; ++batch) {
            for (int i = -options.warmup; i < options.iterations; ++i) {
                Message tag{Kind::ready, case_id, batch, i};
                gather(sockets, tag, options);
                tag.kind = Kind::release;
                for (int socket : sockets) send_message(socket, tag, options);
            }
            const auto means = gather(sockets, Message{Kind::batch, case_id, batch, options.iterations}, options);
            double device_max = -1, wall_max = -1, release_max = -1;
            int device_rank = -1, wall_rank = -1;
            for (int rank = 0; rank < kRanks; ++rank) {
                const auto& row = means[rank];
                ranks << cases[case_id].name << ',' << batch << ',' << rank << ',' << row.device_us
                      << ',' << row.host_wall_us << ',' << row.host_release_us << ',' << row.host_enqueue_us << '\n';
                if (row.device_us > device_max) { device_max = row.device_us; device_rank = rank; }
                if (row.host_wall_us > wall_max) { wall_max = row.host_wall_us; wall_rank = rank; }
                release_max = std::max(release_max, row.host_release_us);
            }
            summaries[case_id].device.push_back(device_max);
            summaries[case_id].wall.push_back(wall_max);
            summaries[case_id].release.push_back(release_max);
            maxima << cases[case_id].name << ',' << batch << ',' << device_max << ',' << device_rank
                   << ',' << wall_max << ',' << wall_rank << ',' << release_max << '\n';
            ranks.flush();
            maxima.flush();
            std::cout << cases[case_id].name << " batch " << batch + 1 << '/' << options.batches
                      << " rank-MAX mean device unit " << device_max << " us\n" << std::flush;
        }
    }
    gather(sockets, Message{Kind::done}, options);
    return summaries;
}

void terminate_children(const std::vector<pid_t>& children) {
    for (pid_t pid : children) if (pid > 0) kill(pid, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (pid_t pid : children) {
        if (pid <= 0) continue;
        int status;
        if (waitpid(pid, &status, WNOHANG) == 0) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
    }
}

void signal_handler(int signal) { interrupted = signal; }

}  // namespace

int main(int argc, char** argv) {
    std::vector<pid_t> children;
    Options options;
    bool created_output = false;
    try {
        options = parse_options(argc, argv);
        fs::create_directories(options.output.parent_path());
        if (!fs::create_directory(options.output)) fail("output directory already exists");
        created_output = true;
        write_run_metadata(options, argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        std::signal(SIGPIPE, SIG_IGN);
        std::array<std::array<int, 2>, kRanks> pairs{};
        for (auto& pair : pairs) {
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair.data()) != 0) fail("socketpair failed");
        }
        for (int rank = 0; rank < kRanks; ++rank) {
            const pid_t pid = fork();
            if (pid < 0) fail("fork failed");
            if (pid == 0) {
                std::signal(SIGINT, SIG_DFL);
                std::signal(SIGTERM, SIG_DFL);
                for (int other = 0; other < kRanks; ++other) {
                    close(pairs[other][0]);
                    if (other != rank) close(pairs[other][1]);
                }
                try {
                    worker(rank, pairs[rank][1], options);
                    close(pairs[rank][1]);
                    std::_Exit(0);
                } catch (const std::exception& error) {
                    try {
                        write_file(options.output / ("rank-" + std::to_string(rank) + ".error.json"),
                                   "{\"rank\":" + std::to_string(rank) + ",\"error\":" + quote(error.what()) + "}");
                    } catch (...) {}
                    std::cerr << "rank " << rank << ": " << error.what() << '\n';
                    std::_Exit(1);
                }
            }
            children.push_back(pid);
        }
        std::vector<int> sockets;
        for (auto& pair : pairs) { close(pair[1]); sockets.push_back(pair[0]); }
        const auto summaries = coordinate(sockets, options);
        for (int fd : sockets) close(fd);
        for (pid_t& pid : children) {
            int status = 0;
            pid_t result;
            do { result = waitpid(pid, &status, 0); } while (result < 0 && errno == EINTR && !interrupted);
            pid = 0;
            if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) fail("worker exited unsuccessfully");
        }
        write_summary(options, summaries);
        std::cout << "Results: " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        terminate_children(children);
        std::cerr << error.what() << '\n';
        if (created_output) {
            try { write_file(options.output / "failure.json", "{\"error\":" + quote(error.what()) + "}"); }
            catch (...) {}
        }
        return 1;
    }
}
