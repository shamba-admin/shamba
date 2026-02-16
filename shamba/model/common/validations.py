from typing import List
from marshmallow import ValidationError


def validate_between_0_and_1(value: List[float]) -> None:
    errors = [
        f"Element at index {i} ({x}) is not between 0 and 1."
        for i, x in enumerate(value)
        if not 0 <= x <= 1
    ]
    if errors:
        raise ValidationError(errors)

def validate_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return "Please enter an integer"

def validate_numerical(value):
    try:
        float(value)
        return True
    except ValueError:
        return "Please enter a valid number."
    
def validate_positive_numerical(value):
    if value > 0:
        return True
    else:
        return "Please enter a positive number"
    
def validate_positive_or_zero_numerical(value):
    if value >=0:
        return True
    else:
        return "Please enter a non-negative number"
    
def validate_positive_or_zero_numerical_list(value: List[float]) -> None:
    errors = [
        f"Element at index {i} ({x}) is negative."
        for i, x in enumerate(value)
        if validate_positive_or_zero_numerical(x) != True
    ]
    if errors:
        raise ValidationError(errors)
