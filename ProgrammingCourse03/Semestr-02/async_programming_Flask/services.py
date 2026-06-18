import asyncio
import aiohttp
import logging

# Настройка логирования для отслеживания возможных сетевых ошибок в консоли
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://jsonplaceholder.typicode.com"
TIMEOUT_SECONDS = 5.0  # Защита от зависания сетевых запросов

async def fetch_json(session: aiohttp.ClientSession, url: str) -> list | dict | None:
    """
    Универсальная корутина для выполнения асинхронных GET-запросов.
    Обрабатывает таймауты, ошибки API и отсутствие сетевого соединения.
    """
    try:
        async with session.get(url, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                logger.error(f"Ошибка API: {url} вернул статус {response.status}")
                return None
            return await response.json()
    except asyncio.TimeoutError:
        logger.error(f"Превышено время ожидания (таймаут) при запросе к {url}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"Сетевая ошибка при обращении к {url}: {e}")
        return None

async def get_users_by_criteria(session: aiohttp.ClientSession, username: str, email: str) -> list:
    """
    Получает полный список пользователей и фильтрует их по подстроке.
    Регистр поиска не учитывается.
    """
    if not username and not email:
        return []
    
    users = await fetch_json(session, f"{BASE_URL}/users")
    if not users:
        return []
        
    filtered_users = []
    for user in users:
        # Проверка вхождения подстроки без учета регистра
        match_username = not username or username.lower() in user.get('username', '').lower()
        match_email = not email or email.lower() in user.get('email', '').lower()
        
        if match_username and match_email:
            filtered_users.append(user)
            
    return filtered_users

async def get_posts_by_criteria(session: aiohttp.ClientSession, title: str, body: str) -> list:
    """
    Получает полный список постов и фильтрует по title и body (содержимому текста).
    """
    if not title and not body:
        return []
        
    posts = await fetch_json(session, f"{BASE_URL}/posts")
    if not posts:
        return []
        
    filtered_posts = []
    for post in posts:
        match_title = not title or title.lower() in post.get('title', '').lower()
        match_body = not body or body.lower() in post.get('body', '').lower()
        
        if match_title and match_body:
            filtered_posts.append(post)
            
    return filtered_posts

async def get_user_by_id(session: aiohttp.ClientSession, user_id: int) -> dict | None:
    """Получение данных конкретного пользователя по ID (для автора поста)"""
    return await fetch_json(session, f"{BASE_URL}/users/{user_id}")

async def perform_async_search(username: str, email: str, title: str, body: str) -> dict:
    """
    Главная корутина-координатор, выполняющая параллельные запросы.
    Использует asyncio.gather для конкурентного выполнения.
    """
    async with aiohttp.ClientSession() as session:
        # Очередь задач: ищем пользователей и посты параллельно
        users_task = get_users_by_criteria(session, username, email)
        posts_task = get_posts_by_criteria(session, title, body)
        
        # Конкурентный запуск двух независимых ветвей поиска
        found_users, found_posts = await asyncio.gather(users_task, posts_task)
        
        # Оптимизация получения авторов постов (Решение проблемы N+1 запросов)
        if found_posts:
            # Выделяем только уникальные userId найденных постов
            user_ids = {post['userId'] for post in found_posts}
            
            # Создаем словарь задач, где ключ - ID пользователя, значение - корутина
            author_tasks = {uid: get_user_by_id(session, uid) for uid in user_ids}
            
            # Извлекаем ключи и параллельно запрашиваем информацию о пользователях
            author_ids = list(author_tasks.keys())
            author_results = await asyncio.gather(*[author_tasks[uid] for uid in author_ids])
            
            # Формируем карту соответствий (ID автора -> Его имя)
            authors_map = {}
            for uid, user_data in zip(author_ids, author_results):
                if user_data:
                    authors_map[uid] = user_data.get('name', 'Неизвестный автор')
                else:
                    authors_map[uid] = 'Ошибка загрузки автора'
            
            # Внедряем имя автора непосредственно в структуру каждого поста
            for post in found_posts:
                post['author_name'] = authors_map.get(post['userId'], 'Неизвестный автор')
                
        return {
            'users': found_users,
            'posts': found_posts
        }