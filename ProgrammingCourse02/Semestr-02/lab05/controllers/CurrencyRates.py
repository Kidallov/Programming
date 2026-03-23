import requests
import xml.etree.ElementTree as ET

class CurrencyRates:
    _instance = None

    def __new__(cls, *args, **kwargs): # реализован паттерн Singleton
        if not cls._instance:
            cls._instance = super(CurrencyRates, cls).__new__(cls)
            cls._instance.currencies = ["USD", "EUR", "GBP"]  # По умолчанию
        return cls._instance

    def get_rates(self):
        url = "https://www.cbr.ru/scripts/XML_daily.asp"
        response = requests.get(url)
        if response.status_code == 200:
            # Парсинг XML и получение курсов валют
            rates = {}
            tree = ET.ElementTree(ET.fromstring(response.text))
            root = tree.getroot()

            for child in root.findall('Valute'):

                currency_id = child.find('CharCode')
                rate = child.find('Value')

                if currency_id is not None and rate is not None:
                    currency_id = currency_id.text
                    rate = float(rate.text.replace(',', '.'))
                    if currency_id in self.currencies:
                        rates[currency_id] = rate
            return rates
        else:
            raise Exception("Ошибка при получении данных")

    def set_currencies(self, currencies): # реализован сеттер
        self.currencies = currencies

    def get_tracked_currency_info(self): # реализован геттер
        rates = self.get_rates()
        info = {}
        for currency in self.currencies:
            rate = rates.get(currency)
            info[currency] = rates.get(currency, "Не найден")
        return info



