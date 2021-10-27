# Simple declarative validation.

"""
Simple declarative validation lib.

From: https://blog.drewolson.org/declarative-validation
"""
from dataclasses import dataclass
from functools import reduce
from toolz import curry
from typing import Any


@dataclass
class Valid:
    value: Any

    def is_valid(self):
        return True

    def apply(self, other):
        if other.is_valid():
            return Valid(self.value(other.value))
        else:
            return other

    def and_then(self, f):
        return f(self.value)


@dataclass
class Invalid:
    value: Any

    def is_valid(self):
        return False

    def apply(self, other):
        if other.is_valid():
            return self
        else:
            return Invalid(self.value + other.value)

    def and_then(self, f):
        return self


def validate_into(f, *args):
    return reduce(lambda a, b: a.apply(b), args, Valid(curry(f)))
