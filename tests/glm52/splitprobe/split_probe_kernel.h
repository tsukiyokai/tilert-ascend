#ifndef TESTS_GLM52_SPLITPROBE_SPLIT_PROBE_KERNEL_H
#define TESTS_GLM52_SPLITPROBE_SPLIT_PROBE_KERNEL_H

void split_probe_launch(void* stream, void* source, void* high_out,
                        void* low_out, void* trace, void* packed);

#endif
