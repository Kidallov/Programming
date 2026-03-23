PARAMS = {'precision': 0.00001, 'dest': 'output.txt'}

def load_params(file="params.ini"):
    """Загрузка параметров из файла"""
    global PARAMS
    try:
        with open(file, mode='r', errors='ignore') as f:
            lines = f.readlines()
            for l in lines:
                param = l.split('=')
                param[1] = param[1].strip('\n')
                if param[0] != 'dest':
                    try:
                        param[1] = eval(param[1])  # Безопасность eval можно улучшить
                    except Exception:
                        print(f"Ошибка преобразования параметра {param[0]}: {param[1]}")
                        continue
                PARAMS[param[0]] = param[1]
    except FileNotFoundError:
        print(f"Файл настроек {file} не найден. Используются параметры по умолчанию.")
    except PermissionError:
        print(f"Недостаточно прав для чтения файла {file}. Используются параметры по умолчанию.")
    else:
        print(f"Параметры успешно загружены из {file}.")
    finally:
        return PARAMS

def write_log(*args, action, file):
    if not isinstance(file, str) or not file.endswith('.txt'):
        raise Exception(f"Ошибка записи в файл {file}. Записать не удалось.")
    # Допустим, функция записывает сумму чисел в файл
    try:
        result = sum(args) if action == 'sum' else None
        with open(file, 'w') as f:
            f.write(str(result))
    except Exception as e:
        raise Exception(f"Ошибка записи в файл {file}: {str(e)}")


def calculate(*args, **kwargs):
    """
    Основная функция для вычисления
    """
    print("Загружаем параметры...")
    try:
        params = load_params('params.ini')
        print(f"Параметры: {params}")
    except Exception as e:
        print(f"Ошибка загрузки параметров: {e}")
        print("Используются параметры по умолчанию.")

    # Выполняем расчет
    try:
        res = sum(args)  # Здесь может быть любое вычисление
        print(f"Результат вычисления: {res}")
    except TypeError as e:
        print(f"Ошибка вычисления: {e}")
        return

    # Пытаемся записать результат в лог
    try:
        write_log(*args, action='sum', result=res, file=PARAMS.get('dest', 'output.txt'))
    except Exception as e:
        print(f"Ошибка записи лога: {e}")
        print(f"Данные для лога: sum: {args} = {res}")

# Тестовый вызов
if __name__ == "__main__":
    print("Перед тестом:")
    print(PARAMS)

    # Пример теста
    calculate(1, 2, 3, 4, 5)
