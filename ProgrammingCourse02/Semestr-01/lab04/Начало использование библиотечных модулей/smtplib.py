import smtplib
from email.mime.text import MIMEText

# Данные для входа и адрес отправителя
smtp_server = "smtp.mail.ru"
port = 465
login = "xxxkidalov@mail.ru"
password = ""

# Создаем сообщение
sender_email = login
receiver_email = "marishik2109@mail.ru"
message = MIMEText("Привет! Ты такой молодец!")
message['Subject'] = "Тестовое сообщение"
message['From'] = sender_email
message['To'] = receiver_email

# Отправляем письмо
try:
    # Подключаемся к серверу
    server = smtplib.SMTP_SSL(smtp_server, port)
    server.login(login, password)

    # Отправляем письмо
    server.sendmail(sender_email, receiver_email, message.as_string())
    print("Письмо успешно отправлено!")
except Exception as e:
    print(f"Ошибка при отправке письма: {e}")
finally:
    server.quit()
