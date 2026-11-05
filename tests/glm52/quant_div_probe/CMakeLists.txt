cmake_minimum_required(VERSION 3.16)
project(quant_div_probe LANGUAGES CXX)

include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(quant_div_probe SHARED quant_div_probe.cpp)
