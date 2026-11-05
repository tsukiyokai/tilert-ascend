
add_ops_compile_options(MoeGatingTopK OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations)
add_ops_compile_options(AddRmsNormBias OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
add_kernels_compile()
