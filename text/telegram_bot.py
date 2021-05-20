import requests

def telegram_bot_sendtext(bot_message):
    bot_token = '1727154835:AAFpb9ZFwD0SAaUyyZ-wmDEVkKSoF4rqXVI'
    bot_chatID = '406153563'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()
