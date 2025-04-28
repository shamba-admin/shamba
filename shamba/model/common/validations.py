from typing import List

def validate_between_0_and_1(value: List[float]) -> List[str]:
    return [
        f"Element at index {i} ({x}) is not between 0 and 1."
        for i, x in enumerate(value)
        if not 0 <= x <= 1
    ]