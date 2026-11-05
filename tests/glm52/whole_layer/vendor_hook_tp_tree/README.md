# Three-rank layer with FP32 tree reductions

This fixture compares the complete indexed/shared layer pair on three A3 dies.
The reference keeps the original input norm and quantization, native attention,
complete FFN and explicit `BF16.float -> HCCL SUM -> BF16` reductions. The
candidate uses one layer launch containing both FP32 tree reductions. Both
arms retain the explicit Top-K copy and the next layer's actual output/residual
dependencies. Parameters are synthetic TP16-shaped shards; thirteen CPU peers
provide constructor geometry without initializing an NPU.

The HCCL environment selects AIV deterministic execution. Its three-rank oracle
is `(partial[0].float() + partial[2].float()) + partial[1].float()`, with one final
BF16 conversion. Each rank has different sharded matrices and identical
replicated parameters. Rank 1 and rank 2 rotate the original matrices by one
and two rows respectively. All three parameter and output witnesses are checked.

The eager and official ACL graph variants use one capacity-32 communication
window across shapes and changing inputs. The window is 2,359,744 bytes per
rank and advances four device epochs per two-layer execution. Host cache reset
and poison do not touch it. Graphs are reset before the final rank rendezvous,
window release and SHMEM finalization. Capture must leave all 113 observed
boundaries unchanged; replay must execute without returning to Python.

`stage.py` verifies the accepted two-rank fixture, derives the three-rank
communicators and literal FP32 reference, and records every source difference.
The original layer arithmetic, parameter postprocessing, NZ descriptor repair,
graph implementation and parent validators remain in the fixture. CPU checks
exercise cancellation, zero signs, final rounding, third-rank identity, reference
collective dtype, communicator call order, and a real four-process Gloo world
whose last rank enters the HCCL nonmember path without initializing an NPU.

The initial device scope is M1 and M12, where the installed HCCL AIV tree has
already been observed with three ranks. Other shapes, TP16 execution and model
TPOT require their own runtime selection and device comparisons.
