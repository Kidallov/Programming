import logging
import math


def convert_precision(tolerance):
    return abs(int(round(math.log10(tolerance))))


def log_calculation(func):
    def wrapper(*args, action, tolerance=1e-6):
        logging.info(f"Calculating with args: {args} and action: {action}")
        result = func(*args, action=action, tolerance=tolerance)
        logging.info(f"Result: {result}")
        return result

    return wrapper


@log_calculation
def calculate(*args, action, tolerance=1e-6):
    if action in ['+', '-', '*', '/']:
        if action == '+':
            result = sum(args)
        elif action == '-':
            result = args[0] - sum(args[1:])
        elif action == '*':
            result = math.prod(args)
        elif action == '/':
            result = args[0]
            for num in args[1:]:
                if num == 0:
                    return "Error!"
                result /= num
        return round(result, convert_precision(tolerance))

    if action == 'medium':
        return round(sum(args) / len(args), convert_precision(tolerance))
    elif action == 'variance':
        mean = sum(args) / len(args)
        variance = round(sum((x - mean) ** 2 for x in args) / len(args), convert_precision(tolerance))
        return variance
    elif action == 'std_deviation':
        return round(math.sqrt(calculate(*args, action='variance')), convert_precision(tolerance))
    elif action == 'median':
        sorted_args = sorted(args)
        mid = len(sorted_args) // 2
        return round((sorted_args[mid - 1] + sorted_args[mid]) / 2, convert_precision(tolerance)) if len(
            sorted_args) % 2 == 0 else round(sorted_args[mid], convert_precision(tolerance))
    elif action == 'q3_q1':
        sorted_args = sorted(args)
        q1 = sorted_args[len(sorted_args) // 4]
        q3 = sorted_args[(3 * len(sorted_args)) // 4]
        return round(q3 - q1, convert_precision(tolerance))

    return "Invalid action!"


def calculator():
    while True:
        input_data = input(
            "Enter numbers separated by spaces and an operator (e.g., '+', 'medium', 'variance'): ").split()
        if len(input_data) < 2:
            print("Error! Please enter at least one number and an operator.")
            continue

        action = input_data[-1]
        numbers = list(map(float, input_data[:-1]))
        result = calculate(*numbers, action=action)
        print(f"Result: {result}")

        if input("Do you want to perform another operation? (Type 'no' to exit): ") == "no":
            break


if __name__ == "__main__":
    calculator()
