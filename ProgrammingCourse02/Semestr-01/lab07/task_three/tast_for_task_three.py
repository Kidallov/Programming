from hypothesis import given, strategies as st
import pytest
from task_three import factorial

@given(st.integers(min_value=0, max_value=100))
def test_factorial_correctness(n):
    if n == 0 or n == 1:
        assert factorial(n) == 1
    else:
        result = factorial(n)
        expected = 1
        for i in range(1, n + 1):
            expected *= i
        assert result == expected

@given(st.integers(min_value=0, max_value=99))
def test_factorial_property(n):
    """Проверяет свойство факториала: (n + 1)! = (n + 1) * n!."""
    assert factorial(n + 1) == (n + 1) * factorial(n)

@given(st.integers(min_value=0, max_value=100))
def test_factorial_non_negative(n):
    """Проверяет, что факториал всегда возвращает неотрицательное значение."""
    assert factorial(n) >= 0

@given(st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers(max_value=-1)))
def test_factorial_invalid_input(n):
    """Проверяет, что функция выбрасывает ValueError для некорректных входных данных."""
    with pytest.raises(ValueError):
        factorial(n)

