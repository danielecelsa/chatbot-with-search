import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

# ---------- background loop management ----------
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_thread: Optional[threading.Thread] = None
_background_lock = threading.Lock()


def _start_background_loop_if_needed() -> asyncio.AbstractEventLoop:
    """Starts the background asyncio event loop if it's not already running."""
    global _background_loop, _background_thread
    with _background_lock:
        if _background_loop is not None and _background_loop.is_running():
            return _background_loop
        loop = asyncio.new_event_loop()

        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run_loop, args=(loop,), daemon=True, name="bg-asyncio-loop")
        t.start()
        _background_loop = loop
        _background_thread = t
        return _background_loop


def run_coro_sync(coro, timeout: Optional[float] = None) -> Any:
    """
    Run a coroutine on the persistent background loop and block until it finishes.
    Returns the coroutine result or raises an exception.
    """
    loop = _start_background_loop_if_needed()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout)
    except FutureTimeoutError:
        future.cancel()
        raise


# ---------- collecting events (blocking) ----------
def collect_events_from_agent(agent, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
                              timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Run agent.astream(...) inside the background loop and collect all events into a Python list.
    """
    async def _collect():
        items = []
        # The 'version' parameter is often not needed for astream, simplifying the call.
        async for ev in agent.astream(inputs, config=config, stream_mode="updates"):
            items.append(ev)
        return items

    return run_coro_sync(_collect(), timeout=timeout)
