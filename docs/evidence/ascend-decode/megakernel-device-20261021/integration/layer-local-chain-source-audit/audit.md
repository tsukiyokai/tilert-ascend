# Local layer phase contract

The frozen 609c973e source has no identified phase or buffer-lifetime counterexample under its parent-validated, disjoint GM ownership contract. This audit does not establish TP correctness.

Input norm keeps its internal mixed join. The caller then drains all pipelines, publishes DDR, joins all mixed cores, and resets the pipe before attention. Indexed and shared attention each retain their final EndPhase after MLA-tail publication, so their local objects have left scope before the FFN bridge allocates UB. The input norm retained residual remains a separate GM tensor throughout attention; FFN receives that residual and the attention partial. The bridge retains all three original EndPhase calls. The FFN retains its five body joins and terminal PIPE_ALL/DDR; it has no later in-kernel consumer.

The attention MLA-tail header is included first. Its complete text differs from the FFN copy only at the standalone Forward dequantization instantiation. Their shared declarations and all helper bodies match; FFN shared.Store explicitly uses CombinedScales, so include order does not change its scale order. The dequant header is only an include shim.

The first SetSysWorkspace selects the 16 MiB bridge workspace for the entire launch. Installed CANN 9.2 ignores subsequent non-force selection attempts. The norm phase uses no Matmul workspace; later attention and FFN phases are separated by their existing joins. The GM owner must remain live through completion. See the adjacent source identities and the existing ffn-system-workspace-source-audit for the installed SDK evidence.

The raw host C ABI checks producer address equalities and geometry, not every tensor's allocation extent or aliasing. The LayerCase fixture first calls the real attention and FFN Torch validators, retains all owners, and allocates phase arenas separately. Calling this C ABI without those conditions is outside the checked interface.

AR2 cannot be inserted immediately after the current FFN terminal publication alone. Before any same-kernel communication consumer or UB reuse, all cores must execute PIPE_ALL, DDR publication, a mixed join, and Reset. A kernel task completion boundary is not a callable phase boundary.
