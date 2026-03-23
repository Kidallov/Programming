import requests
from xml.etree import ElementTree

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class CurrencyRates(metaclass=SingletonMeta):
    URL = "https://www.cbr.ru/scripts/XML_daily.asp"
    CODES = {"USD": "R01235", "EUR": "R01239", "GBP": "R01035"}

    def __init__(self, char_codes =  ['USD', 'EUR', 'GBP']):
        self._rates = {}
        self._char_codes = char_codes
        self._ids = {}

        self._load_currency_ids()
        self._fetch_rates()

    @property
    def rates(self):
        return self._rates

    @property
    def char_codes(self):
        return self._char_codes

    def _load_currency_ids(self):
        response = requests.get(self.URL)
        if response.status_code != 200:
            raise ConnectionError("Не удалось получить данные")

        tree = ElementTree.fromstring(response.content)
        available_ids = {}
        for valute in tree.findall('Valute'):
            char_code = valute.find('CharCode').text
            val_id = valute.attrib['ID']

            if char_code in self.char_codes:
                available_ids[char_code] = val_id

        missing = set(self.char_codes) - set(available_ids.keys())
        if missing:
            raise ValueError(f"Следующие валюты не найдены в XML: {', '.join(missing)}")

        self._ids = available_ids

    def _check_char_codes(self):
        response = requests.get(self.URL)
        if response.status_code == 200:
            tree = ElementTree.fromstring(response.content)
            available_codes = []
            for _code in tree.findall('.//CharCode'):
                available_codes.append(_code.text)

            return all(list(map(lambda _code : _code in available_codes, self._char_codes)))

        else:
            raise ConnectionError("Не удалось получить данные с сайта ЦБ РФ")

    def _fetch_rates(self):
        response = requests.get(self.URL)
        if response.status_code != 200:
            raise ConnectionError("Не удалось получить данные с сайте ЦБ")

        tree = ElementTree.fromstring(response.content)

        for char_code, val_id in self._ids.items():
            element = tree.find(f".//Valute[@ID='{val_id}']/Value")
            if element is not None:
                rate = float(element.text.replace(",", "."))
                self._rates[char_code] = round(rate, 4)

if __name__ == "__main__":
    rates = CurrencyRates()

    print(rates.rates)  # Вывод всех курсов
    print("Курс USD:", rates.rates.get("USD"))

# Использование декораторов для реализации паттерна "Одиночка" может привести к проблемам с состоянием, гибкостью, отладкой и непредсказуемым поведением.