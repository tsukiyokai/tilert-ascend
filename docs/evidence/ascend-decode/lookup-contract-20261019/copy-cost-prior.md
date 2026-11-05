# Copy-source routes as a cost prior

For a fixed current context and flat proposal, changing its source occurrence
does not change target hidden states, expert routes or verification cost.
Prompt-position routes could only predict that cost; they must not constrain
the target's natural expert selection.

[EcoSpec](https://arxiv.org/html/2607.12696v1#S4) uses a trained,
context-conditioned expert predictor. [EVICT](https://arxiv.org/html/2605.00342v2#S3)
combines draft probabilities with an offline verification-cost curve.
[SpecFit](https://arxiv.org/html/2610.15417v1#S3.SS3) uses a target-derived route
atlas in its experiments; acquiring that atlas is outside its timing.
These results motivate cost-aware proposals, without establishing the
predictive value of routes from copied prompt positions on GLM-5.2/A3.

Keep flat widths 8/16/32 as the control. Before adding a policy, test:

1. Source-position routes predict actual expert unions better than each
   layer's frequent experts and the latest accepted routes.
2. The prediction improves cost ordering beyond a static width-cost table;
   validate representative route counts with the existing single-die GMMs.
3. Actual acceptance and measured cost leave enough selection benefit to
   pay for data collection, lookup and the decision itself.

Freeze proposals before reading successor labels. Include pending, padding
and rejected-tail rows, and evaluate held-out requests. Capture prompt routes
during an already-required prefill run. Existing prefix files contain KV and
index state, not those routes; no accuracy or speedup is measured for this
prior yet. Raw uint8 IDs for 75 layers ×8192 positions ×8 experts occupy
4.69 MiB, excluding metadata and capture overhead.

[MoE-Spec](https://arxiv.org/html/2602.16052v1#S3.SS3) caps the verifier's
expert set by truncating or substituting natural routes. That changes the
target computation; its reported throughput does not establish a lossless
verification optimization under this project's frozen-teacher criterion.
