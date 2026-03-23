import sys
import json
import os
import sqlite3
import logging
import gc
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

def trace(func=None, *, handle=sys.stdout):
    if func is None:
        return lambda func: trace(func, handle=handle)

    @wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        log_entry = f"{datetime.now().isoformat()} | {func.__name__} | {args} {kwargs} | {result}\n"
        handle.write(log_entry)
        return result

    return inner


def log_to_json(file_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            log_entry = {
                "datetime": datetime.now().isoformat(),
                "func_name": func.__name__,
                "params": [str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()],
                "result": str(result)
            }

            if file_name.endswith(".json"):
                if os.path.exists(file_name):
                    with open(file_name, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            pass
                data.append(log_entry)
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

            return result

        return wrapper

    return decorator


# Перенос создания таблицы в отдельную функцию
def create_log_table(db_connection: sqlite3.Connection):
    cur = db_connection.cursor()
    # language = SQL
    cur.execute(""" 
        CREATE TABLE IF NOT EXISTS logtable (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            datetime TEXT, 
            func_name TEXT, 
            params TEXT, 
            result TEXT
        )
    """)
    db_connection.commit()


def log_to_sqlite(db_connection):
    # Создание таблицы будет вызываться при первом вызове декоратора
    create_log_table(db_connection)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            cur = db_connection.cursor()
            cur.execute("INSERT INTO logtable (datetime, func_name, params, result) VALUES (?, ?, ?, ?)",
                        (datetime.now().isoformat(), func.__name__, str(args) + str(kwargs), str(result)))
            db_connection.commit()
            return result

        return wrapper

    return decorator


# Вызов функции для создания базы данных
def showlogs(db_connection):
    cur = db_connection.cursor()
    cur.execute("SELECT * FROM logtable")
    logs = cur.fetchall()
    print("\n--- LOGS FROM SQLITE ---")
    for log in logs:
        print(log)
    print("------------------------\n")

@contextmanager
def doc():
    logging.info("Создание in-memory SQLite соединения")
    handle_for_f4 = sqlite3.connect(":memory:")
    try:
        yield handle_for_f4
    finally:
        logging.info("Закрытие соединения с базой данных")
        handle_for_f4.close()
        logging.info("Освобождение памяти через gc.collect()")
        gc.collect()

with doc() as handle:
    @log_to_sqlite(handle)
    def f4(x):
        return x ** 4


    @trace(handle=sys.stderr)
    def increm(x):
        """Инкремент"""
        return x + 1


    @trace(handle=sys.stdout)
    def decrem(x):
        """Декремент"""
        return x - 1

    print(increm.__doc__)
    increm(2)
    decrem(2)
    f4(3)
    f4(5)
    showlogs(handle)

