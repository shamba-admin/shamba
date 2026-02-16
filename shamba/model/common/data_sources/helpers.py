import functools
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def return_none_on_exception(
    *exception_types: type[Exception],
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """
    A decorator that wraps a function to return None if any specified exceptions occur during its execution.
    If no exception types are provided, it defaults to catching all exceptions.

    Args:
        *exception_types (type[Exception]): Variable-length argument list of exception types to catch.

    Returns:
        Callable: A decorated function that returns None if any of the specified exceptions are raised.
    """
    if not exception_types:
        exception_types = (Exception,)  # Catch all exceptions if none specified

    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                print(
                    f"Error occurred in {func.__name__}: {type(e).__name__} - {str(e)}"
                )
                return None

        return wrapper

    return decorator
