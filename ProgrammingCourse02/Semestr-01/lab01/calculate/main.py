def addittion(x, y):
  return x + y

def subtraction(x, y):
  return x - y

def multiplication(x, y):
  return x * y

def division(x, y):
  if y == 0:
      return "Error!"
  return x / y

"""Создали 4 различных функции, которые выполняют разные операции с числами. Также мы учли деление на 0."""

def calculator():
  while True:
      choice = input("Enter choice:\n   1. Addiction \n   2. Subtraction \n   3. Multiplication \n   4. Division \n")

      if choice in ('1', '2', '3', '4'):
          try:
              number_1 = float(input("Enter first number: "))
              number_2 = float(input("Enter second number: "))
          except ValueError:
              print("Invalid input! Please enter a number.")
              continue

          if choice == '1':
              print(f"{number_1} + {number_2} = {addittion(number_1, number_2)}")

          elif choice == '2':
              print(f"{number_1} - {number_2} = {subtraction(number_1, number_2)}")

          elif choice == '3':
              print(f"{number_1} * {number_2} = {multiplication(number_1, number_2)}")

          elif choice == '4':
              result = division(number_1, number_2)
              print(f"{number_1} / {number_2} = {result}")

          if input("Do you want to perform another operation? (Dial 'no' to exit): ") == "no":
              break
      else:
          print("Error! This number is not from the list. Please enter the transaction number.")

"""Основная функция, где мы предлагаем пользователю выбрать операцию, ввести два числа и выводим результат. После операции спрашиваем у пользователя, нужно ли ему что-то еще от нашей программы. Если пользователь вводит 'no', то программа закрывается."""

if __name__ == "__main__":
    calculator()

"""Проверка на главный модуль"""