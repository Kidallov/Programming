from flask import Flask, render_template, request
import asyncio
from services import perform_async_search

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    searched = False
    
    if request.method == 'POST':
        # Получаем данные из полей веб-формы
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        title = request.form.get('title', '').strip()
        body = request.form.get('body', '').strip()
        
        # Проверяем, заполнено ли хотя бы одно поле
        if username or email or title or body:
            # Запускаем асинхронный цикл событий для обработки формы поиска
            results = asyncio.run(perform_async_search(username, email, title, body))
            searched = True
        else:
            # Если все поля пустые, возвращаем пустые структуры
            results = {'users': [], 'posts': []}
            searched = True
            
    return render_template('index.html', results=results, searched=searched)

if __name__ == '__main__':
    # Запуск локального сервера разработки Flask
    app.run(debug=True, port=5000)