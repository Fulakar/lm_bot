# Аргументы для запуска
 - --stt_model' размер модели whisper, по умолчанию 'small'
 - --tts_model', модель с hf для генерации ауидо, по умолчанию 'utrobinmv/tts_ru_free_hf_vits_high_multispeaker'
 - --picovoice_key', ключ [picovoice](console.picovoice.ai)
 - --keyword', ключевое слово для начала работы с ботом, по умолчанию 'bumblebee' (Полный список в pvporcupine.KEYWORDS)
 - --port, порт для запроас в lm studio, по умолчанию '1234'
 - --llm', имя модели в lm studio, по умолчанию 'llama-3.2-3b-instruct'

# HOTWORD DETECT, VOIDE DETECT
Для детекции ключевого слова и используются библиотеки от picovoice поэтому для работы бота надо получить ключи api в [личном кабинете](console.picovoice.ai) для Porcupine и Cobra
# SPEACH TO TEXT
Для транскрибации ауидо используется [whisper](https://github.com/openai/whisper?ysclid=m6auv8gebr439068373) 
# LLM
Запрос к LLM идет через post запрос (наприер в [LM studio](https://lmstudio.ai/), локально поднимающую модель)
# TEXT TO SPEACH
Для озвучивания текста используется [utrobinmv/tts_ru_free_hf_vits_high_multispeaker]([utrobinmv/tts_ru_free_hf_vits_high_multispeaker](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_high_multispeaker)) с huggingface

# Возможности
 - пока только работа с аудио: громкость, переключение, пауза
  
P.S. Лично я поднимаю LLM в LM Studio (в будущем перепишу на TorchServe). Для нормальной работы рекомендую использовать instruct модели llm.
