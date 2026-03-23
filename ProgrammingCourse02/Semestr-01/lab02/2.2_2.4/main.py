import logging

def log_calculation(func):
    def wrapper(x, y, action):
        logging.info(f"Calculating: {x} {action} {y}")
        result = func(x, y, action)
        logging.info(f"Result: {result}")
        return result
    return wrapper

@log_calculation
def calculate(x, y, action):
    if action == '+':
        return addition(x, y)
    elif action == '-':
        return subtraction(x, y)
    elif action == '*':
        return multiplication(x, y)
    elif action == '/':
        return division(x, y)
    else:
        return "Invalid action!"

def addition(x, y):
    return x + y

def subtraction(x, y):
    return x - y

def multiplication(x, y):
    return x * y

def division(x, y):
    if y == 0:
        return "Error!"
    return x / y

def calculator():
    while True:
        print("Enter two numbers and an operator (+, -, *, or /) separated by a space: ")
        input_data = input().split()

        if len(input_data) != 3:
            print("Error! Please enter two numbers and an operator only.")
            continue

        number_1, action, number_2 = input_data

        try:
            number_1 = float(number_1)
            number_2 = float(number_2)
        except ValueError:
            print("Invalid input! Please enter valid numbers.")
            continue

        result = calculate(number_1, number_2, action)
        print(f"Result: {result}")

        if input("Do you want to perform another operation? (Type 'no' to exit): ") == "no":
            break

if __name__ == "__main__":
    calculator()