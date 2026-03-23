import configparser
import os


def read_settings(filename='settings.ini'):
    """Считывает настройки из файла INI.

    Args:
        filename (str): Путь к файлу настроек.

    Returns:
        dict: Настройки в виде словаря.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если файл содержит некорректный формат.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    config = configparser.ConfigParser()
    try:
        config.read(filename)
        return {
            'host': config['database']['host'],
            'port': int(config['database']['port']),
            'user': config['database']['user'],
            'password': config['database']['password']
        }
    except Exception as e:
        raise ValueError(f"Error parsing settings file: {e}")


def write_to_file(filename, line):
    """Записывает строку в файл. Создает файл, если он отсутствует.

    Args:
        filename (str): Имя файла.
        line (str): Строка для записи.
    """
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as file:
        file.write(line + '\n')

