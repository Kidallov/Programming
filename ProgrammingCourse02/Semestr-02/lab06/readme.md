# Отчет по лабораторной работе №5

## Структура файлов проекта:
```
Python/
├── app.py               
├── model.py             
├── controller.py        
├── templates/
│   ├── index.html       
│   └── update.html      
├── database.db          
├── requirements.txt     
```

### Файл `app.py:
```
from flask import Flask
from controller import bp

app = Flask(__name__)
app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True)
```

### Файл `model.py`:
```
import requests
import xml.etree.ElementTree as ET
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class CurrencyRate(Base):
    __tablename__ = 'currency_rates'
    id = Column(String(3), primary_key=True)
    datetime = Column(String(20))
    value = Column(Float)


class CurrencyRatesSingleton:
    _instance = None

    def __new__(cls, db_url="sqlite:///database.db"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db(db_url)
            cls._instance.tracked_currencies = ['USD', 'EUR']
        return cls._instance

    def _init_db(self, db_url):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def fetch_rates(self):
        url = 'https://www.cbr.ru/scripts/XML_daily.asp'
        response = requests.get(url)
        tree = ET.fromstring(response.content)
        now = datetime.now().strftime('%d-%m-%Y %H:%M')
        rates = {}

        for valute in tree.findall('Valute'):
            code = valute.find('CharCode').text
            if code in self.tracked_currencies:
                value = float(valute.find('Value').text.replace(',', '.'))
                rates[code] = {'datetime': now, 'value': value}

        session = self.Session()
        for code, data in rates.items():
            rate = session.query(CurrencyRate).get(code)
            if not rate:
                rate = CurrencyRate(id=code)
                session.add(rate)
            rate.datetime = data['datetime']
            rate.value = data['value']
        session.commit()
        session.close()

    def get_rates(self):
        session = self.Session()
        rates = session.query(CurrencyRate).all()
        session.close()
        return rates

    def set_tracked(self, currency_codes):
        self.tracked_currencies = currency_codes
```

### Файл `controller.py`:
```
from flask import Blueprint, request, redirect, render_template
from model import CurrencyRatesSingleton

bp = Blueprint("main", __name__)
rates_model = CurrencyRatesSingleton()

@bp.route('/')
def index():
    rates_model.fetch_rates()
    rates = rates_model.get_rates()
    return render_template('index.html', rates=rates)

@bp.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        codes = request.form.get('currencies')
        codes = [code.strip().upper() for code in codes.split(',') if code.strip()]
        rates_model.set_tracked(codes)
        return redirect('/')
    return render_template('update.html')
```

### Файл `index.html`:
```
<!doctype html>
<html>
<head><title>Курсы валют</title></head>
<body>
    <h1>Курсы валют</h1>
    <table border="1">
        <tr><th>Валюта</th><th>Курс</th><th>Дата</th></tr>
        {% for r in rates %}
            <tr><td>{{ r.id }}</td><td>{{ r.value }}</td><td>{{ r.datetime }}</td></tr>
        {% endfor %}
    </table>
    <a href="/update">Изменить отслеживаемые валюты</a>
</body>
</html>
```

### Файл `update.html`:
```
<!doctype html>
<html>
<head><title>Обновление валют</title></head>
<body>
    <h1>Введите валюты через запятую</h1>
    <form method="post">
        <input name="currencies" placeholder="USD, EUR, GBP" />
        <button type="submit">Обновить</button>
    </form>
    <a href="/">Назад</a>
</body>
</html>
```

### Фвйл `requirements.txt`:
```
flask
sqlalchemy
requests
```

### Вывод и пояснения

В ходе выполнения лабораторной работы №5 было реализовано веб-приложение на Flask, которое позволяет получать и отображать актуальные курсы валют с сайта Центрального банка России. Основной акцент был сделан на архитектурное разделение проекта и взаимодействие с внешними API и базой данных.

**Что было проведено в данной лабораторной работе:**

1. **Архитектура MVC**:
   Код проекта организован по принципам Model-View-Controller:

   * `model.py` содержит логику работы с данными и базой (модель).
   * `controller.py` обрабатывает маршруты и взаимодействует с моделью (контроллер).
   * `templates/*.html` отображают данные пользователю (представление).

2. **Работа с API**:
   С помощью библиотеки `requests` происходит загрузка XML-данных с сайта ЦБ РФ. Эти данные парсятся и сохраняются в локальную базу.

3. **Хранение данных**:
   Для хранения используется SQLite через SQLAlchemy ORM, что упрощает работу с базой данных и позволяет легко модифицировать структуру при необходимости.

4. **Паттерн Singleton**:
   Класс `CurrencyRatesSingleton` реализован по паттерну одиночка, что обеспечивает единый доступ к модели данных во всем приложении.

5. **Динамическое обновление данных**:
   Пользователь может сам указать интересующие валюты, и система начнёт отслеживать только их. Это повышает гибкость приложения.

6. **Работа с шаблонами Flask**:
   Используются шаблоны Jinja2, позволяющие удобно отображать данные на HTML-страницах с минимальным дублированием кода.
