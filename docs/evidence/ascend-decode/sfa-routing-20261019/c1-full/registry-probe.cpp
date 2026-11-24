#include "base/registry/op_impl_space_registry_v2.h"
#include <fstream>
#include <iostream>
#include <string>
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  gert::OpImplSpaceRegistryV2 registry;
  gert::OppSoDesc desc({ge::AscendString(argv[1])}, ge::AscendString("codex_sfa_host_audit"));
  const auto status = registry.AddSoToRegistry(desc);
  if (status != ge::GRAPH_SUCCESS) { std::cerr << "AddSoToRegistry=" << status << "\n"; return 3; }
  std::ifstream names(argv[2]);
  std::string name;
  bool missing = false;
  while (std::getline(names, name)) {
    const auto *impl = registry.GetOpImpl(name.c_str());
    std::cout << name << '\t' << (impl != nullptr) << '\t'
              << (impl != nullptr && impl->tiling != nullptr) << '\t'
              << (impl != nullptr && impl->tiling_parse != nullptr) << '\t'
              << (impl != nullptr ? impl->max_tiling_data_size : 0) << '\n';
    missing |= impl == nullptr;
  }
  return missing ? 4 : 0;
}
