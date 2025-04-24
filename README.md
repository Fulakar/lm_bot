# Структура
call_func.py - функции вызываемые агентом

fulakar_agent.py - агент

app.py - главный файл

---
# Переменные для запуска
- WHISPER_SIZE - размер модели whisper (**faster whisper**), по умолчанию 'small'<br>
- TTS_MODEL - модель с hf для генерации ауидо, по умолчанию 'utrobinmv/tts_ru_free_hf_vits_high_multispeaker'<br>
- PICOVOICE_KEY - ключ [picovoice](console.picovoice.ai)<br>
- KEYWORD - ключевое слово для начала работы с ботом, по умолчанию 'bumblebee' (Полный список в pvporcupine.KEYWORDS)<br>
- PORT = 1234<br>
- HOST = "localhost"<br>
- LLM_NAME = "yandexgpt-5-lite-8b-instruct"<br>
- API_ID, API_HASH - API для библиотеки telethon (не путать с telebot), получать [тут](https://my.telegram.org/auth?to=%3Fspm%3Da2ty_o01.29997173.0.0.7fdfc921LNM7KH)<br>
- ALLOWED_DIALOG - Список разрешенных диалогов (название чата из TG)

---
Функции агента:
- os_music - управление музыкой <br>
- duckduck_search - поисковик <br>
- send_message - отправка сообщение пользователю (telegram)(**работает только для диалогов из ALLOWED_DIALOG**) <br>
- get_unread_dialogs - наличие непрочитанных сообщений (telegram)(**работает только для диалогов из ALLOWED_DIALOG**) <br>

Вспомогательные функции:
- HOTWORD DETECT, VOIDE DETECT - для детекции ключевого слова и используются библиотеки от picovoice поэтому для работы бота надо получить ключи api в [личном кабинете](console.picovoice.ai) для Porcupine и Cobra<br>
- SPEACH TO TEXT - для транскрибации ауидо используется [faster-whisper](https://github.com/SYSTRAN/faster-whisper?ysclid=m9v0ctkaih604284609)<br>
- TEXT TO SPEACH - для озвучивания текста используется [utrobinmv/tts_ru_free_hf_vits_high_multispeaker]([utrobinmv/tts_ru_free_hf_vits_high_multispeaker](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_high_multispeaker)) с huggingface<br>
