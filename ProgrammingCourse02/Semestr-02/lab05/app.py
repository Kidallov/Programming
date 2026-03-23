# Импортируем модули
from flask import Flask
from controllers.CurrencyRates import CurrencyRates
from controllers.DatabaseController import DatabaseController
from controllers.ViewController import ViewController

app = Flask(__name__)

# Объявляем, что будет отображаться по корневому маршруту
@app.route('/')
def index():

    # Инициализация классов
    currency_rates = CurrencyRates()
    db_controller = DatabaseController()
    view_controller = ViewController()

    # Получение курсов валют и сохранение их в БД
    rates = currency_rates.get_rates()
    for currency, rate in rates.items():
        db_controller.save_rate(currency, rate)

    # Передача данных в шаблон
    context = {
        'rates': db_controller.get_rates()
    }
    return view_controller.render('rates_template.html', context)

if __name__ == "__main__":
    app.run(debug=True)
