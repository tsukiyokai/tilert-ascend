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
