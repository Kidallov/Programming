import pytest

def fact(n):
    """
    Функция для вычисления факториала числа n.
    Выбрасывает исключение ValueError, если n < 0.
    """
    if int(n) < 0:
        raise ValueError('Значение должно быть больше либо равно нулю')
    elif int(n) == 0:
        return 1  # Факториал 0 равен 1
    else:
        result = 1
        for i in range(1, int(n) + 1):
            result *= i
        return result

def test_neg_values_fact():
    """Проверяем, что факториал от отрицательных чисел вызывает ValueError."""
    with pytest.raises(ValueError):
        fact(-5)

def test_neg_values_fact_ex_text():
    """Проверяем, что исключение содержит правильный текст."""
    with pytest.raises(ValueError, match="больше либо равно нулю"):
        fact(-3)

def test_zero_fact():
    """Проверяем, что факториал 0 равен 1."""
    assert fact(0) == 1

def test_positive_fact():
    """Проверяем факториал для положительных чисел."""
    assert fact(5) == 120  # 5! = 1*2*3*4*5 = 120
    assert fact(3) == 6    # 3! = 1*2*3 = 6
    assert fact(1) == 1    # 1! = 1

def test_large_number_fact():
    """Проверяем факториал для большого числа."""
    assert fact(10) == 3628800  # 10! = 1*2*3*...*10

if __name__ == "__main__":
    pytest.main(["-v"])
