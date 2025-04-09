import functools
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def return_none_on_exception(
    *exception_types: type[Exception],
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
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
