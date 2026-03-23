from main import addittion, subtraction, multiplication, division

def test_calculator_functions():
  # Тест для функции сложения
  assert addittion(2, 3) == 5

  # Тест для функции вычитания
  assert subtraction(5, 3) == 2

  # Тест для функции умножения
  assert multiplication(4, 5) == 20

  # Тест для функции деления
  assert division(10, 2) == 5
  assert division(5, 0) == "Error!"


  print("Все выполнено успешно!")


if __name__ == "__main__":
  test_calculator_functions()