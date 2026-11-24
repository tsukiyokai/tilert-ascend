"""Deliver decode-thread tokens and retain its request until it exits."""

import asyncio
import queue
import threading


def finish_worker(worker, cleanup):
    """Keep request ownership until the producer has actually terminated."""
    if not worker.is_alive():
        cleanup()
        return None

    def drain():
        worker.join()
        cleanup()

    waiter = threading.Thread(target=drain, name="pd-drain", daemon=True)
    waiter.start()
    return waiter


class TokenQueue:
    """Bind on the response loop before starting its producer thread."""

    def __init__(self):
        self._queue = queue.Queue()
        self._loop = None
        self._event = None

    def bind_loop(self):
        if self._loop is not None:
            raise RuntimeError("The token queue already has a response loop")
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()

    def put(self, item):
        self._queue.put(item)
        self._loop.call_soon_threadsafe(self._event.set)

    def get_nowait(self):
        return self._queue.get_nowait()

    async def get(self, timeout=600):
        async with asyncio.timeout(timeout):
            while True:
                # Clear before inspecting the queue, so a concurrent put
                # either supplies the item or leaves a wakeup for wait().
                self._event.clear()
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    await self._event.wait()
