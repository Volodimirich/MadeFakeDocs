import os
import sys
import argparse
# import logging.config
import telebot
import torch

from modules.model import get_tokenizer, get_model
from modules.engine import predict
from telebot import types


# logging.config.dictConfig(config)
# logger = logging.getLogger('bot')

TBOT_TOKEN = '6212919521:AAGnlqjlfWTVBPD7otysxGiSDw1vHdEDh1Y'
bot = telebot.TeleBot(TBOT_TOKEN)
local_path = None
user_settings = {}
# logger.info("Bot loaded, token = %s", os.environ['TBOT_TOKEN'])



def get_result(message, chat_id):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_name = user_settings[chat_id]['model']
    words = user_settings[chat_id]['word_count']

    model = get_model(model_name, device, local_path, False)
    tokenizer = get_tokenizer(model_name)
    model.eval()
    model.to(device)
    
    predict_config = {'do_sample': True, 'num_beams': 2, 'temperature': 1.5, 
              'top_p': 0.9}
    model_inputs = tokenizer(text=message, return_tensors='pt')
    model_inputs = model_inputs.to(device)

    out = model.generate(**model_inputs,
                        #  pad_token_id=tokenizer.eos_token_id,
                         do_sample=predict_config['do_sample'],
                         num_beams=predict_config['num_beams'],
                         temperature=predict_config['temperature'],
                         top_p=predict_config['top_p'],
                         max_length=words,
                         )

    generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)
    user_settings[chat_id]['mode'] = 'input'
    return generated_text

def init_user(id_val):
    if id_val not in user_settings:
        user_settings[id_val] = {}
        user_settings[id_val]['model'] = 'gpt2'
        user_settings[id_val]['word_count'] = 100
        user_settings[id_val]['mode'] = 'input'


@bot.message_handler(commands=['start'])
def send_welcome(message):
    init_user(message.chat.id)
    bot.reply_to(message, 'Hi there, I am FakeDocsBot. Please write text to start.')
    # logger.info('Sending hello message, chat id - %i', message.chat.id)
    
# Обработчик команды /settings
@bot.message_handler(commands=['settings'])
def settings(message):
    init_user(message.chat.id)
    # Создание объекта клавиатуры
    keyboard = types.InlineKeyboardMarkup(row_width=1)

    # Создание кнопок для модели, количества слов и текущих параметров
    button_model = types.InlineKeyboardButton('Выбор модели', callback_data='model')
    button_word_count = types.InlineKeyboardButton('Количество слов', callback_data='word_count')
    button_current_settings = types.InlineKeyboardButton('Текущие параметры', callback_data='current_settings')

    # Добавление кнопок на клавиатуру
    keyboard.add(button_model, button_word_count, button_current_settings)

    # Отправка сообщения с клавиатурой
    bot.send_message(message.chat.id, 'Выберите настройку:', reply_markup=keyboard)


# Обработчик нажатия Inline кнопок для выбора модели
@bot.callback_query_handler(func=lambda call: call.data.startswith('model_'))
def handle_model_choice(call):
    chat_id = call.message.chat.id
    init_user(chat_id)
    model = call.data.split('_')[1]
    model_matcher = {'gpt2': 'gpt2', 'sber' :'sberbank-ai/rugpt3large_based_on_gpt2',
                     't5-small': 'flan-t5-small', 't5-big': 'flan-t5-large'}


    # Сохранение выбранной модели в user_settings
    user_settings[chat_id]['model'] = model_matcher[model]
    bot.send_message(chat_id, f'Выбрана модель: {model_matcher[model]}')
    user_settings[chat_id]['mode'] = 'input'


# Обработчик нажатия Inline кнопок
@bot.callback_query_handler(func=lambda call: True)
def handle_inline_button_click(call):
    chat_id = call.message.chat.id
    init_user(chat_id)

    if call.data == 'model':
        user_settings[chat_id]['mode'] = 'settings'
        # Отправка сообщения с вариантами выбора модели
        bot.send_message(chat_id, 'Выберите модель:', reply_markup=get_model_keyboard())

    elif call.data == 'word_count':
        user_settings[chat_id]['mode'] = 'settings'
        # Отправка сообщения с запросом ввода количества слов
        bot.send_message(chat_id, 'Введите количество слов (от 10 до 1023):')


    elif call.data == 'current_settings':
        # Отправка текущих параметров пользователя
        if chat_id in user_settings:
            current_settings = user_settings[chat_id]
            bot.send_message(chat_id, f'Текущие параметры:\nМодель: {current_settings["model"]}\nКоличество слов: {current_settings["word_count"]}')
        else:
            bot.send_message(chat_id, 'Текущие параметры не найдены.')

# Обработчик ввода текста сообщения
@bot.message_handler(func=lambda message: True)
def handle_text_input(message):
    chat_id = message.chat.id
    init_user(chat_id)
    print(user_settings)
    setting = user_settings[chat_id]['mode']
    if setting == 'settings':
        # Обработка ввода количества слов
        try:
            word_count = int(message.text)
            if word_count >= 10 and word_count <= 1023:
                # Сохранение настройки количества слов
                user_settings[chat_id]['word_count'] = word_count
                bot.send_message(chat_id, f'Выбрано количество слов: {word_count}')
                user_settings[chat_id]['mode'] = 'input'
            else:
                bot.send_message(chat_id, 'Некорректное количество слов.')
        except ValueError:
            bot.send_message(chat_id, 'Некорректный ввод. Введите число.')
    elif setting == 'input':
        bot.reply_to(message, 'Данные получены, ожидайте')
        user_settings[chat_id]['mode'] = 'wait'
        bot.reply_to(message, get_result(message.text, chat_id))

    else:
        bot.send_message(chat_id, 'Сообщение обрабатывается, подождите. Если это ошибка, перезапустите настройки через \settings')

        # Удаление сохраненной настройки после обработки

# Функция для получения Inline Keyboard Markup для выбора модели
def get_model_keyboard():
    # Создание объекта клавиатуры
    keyboard = types.InlineKeyboardMarkup(row_width=2)

    # Создание кнопок для модели
    button_gpt2 = types.InlineKeyboardButton('gpt2', callback_data='model_gpt2')
    button_t5_pretrained = types.InlineKeyboardButton('sber', callback_data='model_sber')
    button_t5_small = types.InlineKeyboardButton('t5_small', callback_data='model_t5-small')
    button_t5_big = types.InlineKeyboardButton('t5_big', callback_data='model_t5-big')
    
    # Добавление кнопок на клавиатуру
    keyboard.add(button_gpt2, button_t5_pretrained, button_t5_small, button_t5_big)

    return keyboard




if __name__ == '__main__':
    bot.infinity_polling()