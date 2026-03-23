# Работа с БД
import sqlite3
from datetime import datetime

class DatabaseController:
    def __init__(self):
        self.__con = sqlite3.connect('data.sqlite3')
        self.__cursor = self.__con.cursor()  # Создаем курсор после соединения
        self.__create_db()

    def __create_db(self):
        # Удаляем таблицу, если она существует, и создаем новую
        self.__cursor.execute("DROP TABLE IF EXISTS currencies")
        self.__cursor.execute("""
            CREATE TABLE IF NOT EXISTS currencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT,
                date TEXT,
                value FLOAT NOT NULL,
                UNIQUE(currency, date)
            )
        """)
        self.__con.commit()

    def save_rate(self, currency, rate):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Вставляем данные или обновляем существующие записи с помощью INSERT OR REPLACE
        self.__cursor.execute("""
            INSERT OR REPLACE INTO currencies (currency, date, value)
            VALUES (?, ?, ?)
        """, (currency, date, rate))
        self.__con.commit()

    def get_rates(self):
        self.__cursor.execute("SELECT * FROM currencies ORDER BY date DESC")
        return self.__cursor.fetchall()

    def close(self):
        if self.__con:
            self.__con.close()
