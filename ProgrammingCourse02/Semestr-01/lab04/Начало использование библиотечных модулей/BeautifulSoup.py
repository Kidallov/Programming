import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

response = requests.get("https://wttr.in/Санкт-Петербург", headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    weather_block = soup.find('pre')

    if weather_block:
        print(weather_block.text.strip())
    else:
        print("Не удалось найти блок с погодой.")
else:
    print(f"Ошибка: не удалось загрузить данные (код {response.status_code}).")
