# K128 normal-scheduler source contract

The six-field record is a candidate supported by the installed constant planner and unchanged normal Matmul consumer. Its timing effect remains unmeasured.

`include/adv_api/matmul/constant_tiling.h:38-86` (dbfaaa6dd2956f3ae671e99a0efc9f54a79e6e48f9e6979fc54f583a6ac667aa): The installed constant API derives K, L1 depths, L0 double-buffer fields and resource sizes.

`include/adv_api/matmul/tiling.h:36-94` (31c650110ac7fe41a8921e403e85402987d80fc866d78c95e4bbcc8acd9d9536): CFG_NORM is the normal INNER_PRODUCT configuration.

`impl/adv_api/detail/matmul/feature_trait/matmul_feature_trait.h:45-51` (c6182edf8433edcbb34593603bf133c5d38bfce808c2a3013087d82bb2b53c28): The forced M/N L0 double-buffer path requires OUTER_PRODUCT, so this normal consumer does not select it.

`impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h:42-59` (9a572553706b94999ddce6fdeb570384050da84bf1021b810a404710691affeb): The runtime dbL0A/dbL0B conjunction controls both L0 operand buffers.

`impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h:63-120` (9a572553706b94999ddce6fdeb570384050da84bf1021b810a404710691affeb): One buffer still has M_MTE1 wait/free and MTE1_M enqueue/dequeue; offset remains zero.

`impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h:49-82` (4e12f4e3eb4085a8e1302d096bf46d0f1066a60714a308d38feca5d2521e8263): Runtime K128 with baseK128 has one outer iteration and exact K128 tail.

`impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:193-240` (10fafe83537d229acebc130ae25064141cac98353d9310887fda4e564df46783): The scheduler loads the current M/N/K tile and uses the original allocate/enqueue/dequeue protocol.

`impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h:279-295` (10fafe83537d229acebc130ae25064141cac98353d9310887fda4e564df46783): Every MMAD returns the L0 credit through Free.

`parent-routed-body-native.h:176-199` (995a7365295c188d0b9b7efd8f72d0e95946ffd3a3db7c59c356d0e4214eb29f): Project preserves original full-K stride, dynamic 2*count rows, IterateAll and End.

`parent-routed-body-native.h:707-740` (995a7365295c188d0b9b7efd8f72d0e95946ffd3a3db7c59c356d0e4214eb29f): The same normal Projection owns Down; active-expert order and shared.Store remain unchanged.

The original runtime `transLength=0` is retained. The API emits 131072 for both accepted controls and the candidate, so the static structure is not copied wholesale. Each original output tile still covers the same elements and full K128 dot; the INT4 accumulation bound is 8192. Counts 17–32 require the second M tile and remain explicit device gates.

A source-level reduction in K-loop iterations does not establish latency gain: the same change removes operand ping-pong. No UB or GM saving is claimed.
