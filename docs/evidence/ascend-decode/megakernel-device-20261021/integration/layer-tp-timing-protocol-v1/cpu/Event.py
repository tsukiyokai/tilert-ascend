class Event(torch_npu._C._NPUEventBase):
    r"""Wrapper around a NPU event.

    NPU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize NPU
    streams.

    The underlying NPU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(cls, enable_timing=enable_timing, blocking=blocking,
                                         interprocess=interprocess, graph_external=False)

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_npu.npu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_npu.npu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = torch_npu.npu.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def recorded_time(self):
        r"""Returns the record time stamp after the event was recorded.
        """
        return super(Event, self).recorded_time()

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``npuEventSynchronize()``: see
            `NPU Event documentation`_ for more info.
        """
        super(Event, self).synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.npu_event)

    def __repr__(self):
        if self.npu_event:
            return '<torch_npu.npu.Event {0:#x}>'.format(self._as_parameter_.value)
        else:
            return '<torch_npu.npu.Event uninitialized>'
