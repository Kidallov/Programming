# Работа с jinja2 для визуализации
from jinja2 import Environment, FileSystemLoader

# Создаем класс, отвечающий за показ
class ViewController:
    def __init__(self, template_folder='templates'):
        self.env = Environment(loader=FileSystemLoader(template_folder)) # указываем откуда брать шаблоны .html

    def render(self, template_name, context):
        template = self.env.get_template(template_name) # берем нужный нам шаблон
        return template.render(context) # возвращается готовая HTML-страница как готовая, со значениями из context
