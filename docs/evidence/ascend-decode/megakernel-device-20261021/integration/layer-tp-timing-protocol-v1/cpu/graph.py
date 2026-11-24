class graph:
    r"""Context-manager that captures NPU work into a :class:`torch.npu.NPUGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        npu_graph (torch.npu.NPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.npu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.npu.NPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.npu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the aclmdlRICaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During npu graph capture, some actions, such as npuMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `aclmdlRICaptureMode`_

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.
    """  # noqa: B950

    default_capture_stream: typing.Optional["torch.npu.Stream"] = None

    def __init__(
        self,
        npu_graph,
        pool=None,
        stream=None,
        auto_dispatch_capture=False,
        capture_error_mode: str = "global",
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.npu.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise RuntimeError("capture stream is None")
        self.stream_ctx = torch.npu.stream(self.capture_stream)
        self.npu_graph = npu_graph
        self.capture_error_mode = capture_error_mode
        self.npu_graph.auto_dispatch_capture = auto_dispatch_capture

    def __enter__(self):
        log.debug("NPUGRAPH Capture device=%s, pool=%s, mode=%s, auto_dispatch=%s",
                  torch.npu.current_device(), self.pool, self.capture_error_mode,
                  self.npu_graph.auto_dispatch_capture)
        torch.npu.synchronize()
        if force_npugraph_gc:
            gc.collect()
        torch.npu.empty_cache()
        torch_npu.npu.host_empty_cache()

        # Stackoverflow seems comfortable with this pattern
        self.stream_ctx.__enter__()
        if self.npu_graph.auto_dispatch_capture:
            self.npu_graph.graph_dispatch_mode.__enter__()
        self.npu_graph.capture_begin(
            *self.pool, capture_error_mode=self.capture_error_mode
        )

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            log.error("NPUGraph: ERROR — capture failed: %s: %s, "
                      "graph_id=%s, stream=%s, pool=%s, device=%s",
                      exc_type.__name__, exc_value,
                      id(self.npu_graph), self.capture_stream,
                      self.pool, torch.npu.current_device())
            log.warning("NPUGRAPH DFX common causes: default stream, CPU ops, multi-device tensors, "
                        "dynamic control flow on tensor values")
        else:
            log.debug("NPUGraph: capture completed, graph_id=%s, device=%s",
                      id(self.npu_graph), torch.npu.current_device())
        self.npu_graph.capture_end()
        if self.npu_graph.auto_dispatch_capture:
            self.npu_graph.graph_dispatch_mode.__exit__(exc_type, exc_value, traceback)
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()
