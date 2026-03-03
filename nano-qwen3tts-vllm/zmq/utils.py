"""Shared ZMQ/multiprocess utilities (e.g. port finding)."""

try:
    import zmq
except ImportError:
    zmq = None


def find_available_port(start_port: int = 9555, max_attempts: int = 1000) -> int:
    """Find an available port starting from start_port (uses ZMQ bind to test).

    Args:
        start_port: Starting port number to check.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number.

    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    if zmq is None:
        raise ImportError("pyzmq is required. Install with: pip install pyzmq")

    test_ctx = zmq.Context()
    try:
        for port in range(start_port, start_port + max_attempts):
            test_socket = None
            try:
                test_socket = test_ctx.socket(zmq.PUB)
                test_socket.setsockopt(zmq.LINGER, 0)
                test_socket.bind(f"tcp://127.0.0.1:{port}")
                test_socket.close()
                return port
            except zmq.error.ZMQError:
                if test_socket:
                    try:
                        test_socket.close()
                    except Exception:
                        pass
                continue
            except Exception:
                if test_socket:
                    try:
                        test_socket.close()
                    except Exception:
                        pass
                continue
    finally:
        test_ctx.term()

    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}"
    )
