import pytest

from main import convert_precision, calculate

def test_convert_precision():
    assert convert_precision(1e-6) == 6
    assert convert_precision(1e-10) == 10

def test_calculate():
    assert calculate(3, 4, '+', 1e-2) == 7.0
    assert calculate(3.14159, 2, '*', 1e-3) == 6.283
    assert calculate(5, 2, '-', 1e-2) == 3.0
    assert calculate(48, 14, '/', 1e-4) == 3.4286