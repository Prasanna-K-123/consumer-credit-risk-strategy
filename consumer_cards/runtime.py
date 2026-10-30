"""Bound native numerical workers for a predictable local research workload."""
from functools import wraps

from threadpoolctl import threadpool_limits

NATIVE_THREAD_LIMIT = 2


def bounded_native_threads(function):
    """Limit loaded BLAS/OpenMP pools for this call, restoring caller settings.

    Create the controller at invocation so libraries loaded after module import
    are included. The context also restores settings if the function raises.
    This is process-level native-pool control, not a per-request concurrency API.
    """
    @wraps(function)
    def bounded(*args, **kwargs):
        with threadpool_limits(limits=NATIVE_THREAD_LIMIT):
            return function(*args, **kwargs)
    return bounded
