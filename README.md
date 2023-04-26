# FakeDocs
Fake Documents Project

# Запуск gpt2
Перед запуском необходимо установить все зависимости, выполнив следующие команды:
```commandline
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


```commandline
pip install -r requirements.txt
```

Логирование метрик осуществляется в `wandb`. Поэтому перед запуском
необходимо авторизироваться, выполнив команду:
```commandline
wandb login
```

## Конфигурирование
Настройка параметров модели осуществляется в файле `configs/train_config.yaml`
Описание параметров:
```
Logging metrics:
* wandb_project --- Название проекта в `wandb` default="MadeFakeDocs"

Model
* tokenizer_name --- Название токенизатора для модели
* model_name_or_path --- Путь до модели на локальным диске или название модели
* path_save_model --- Путь сохранения результата обучения модели

Dataset
* file_path --- Путь к файлу для обучения
* block_size --- 64
* mlm --- Следует ли использовать маскированное языковое моделирование или нет. Если установлено значение "False", метки совпадают с входными данными

TrainingArguments
* pre_trained --- *На данный момент не используется*
* output_dir --- Каталог с результатами обучения
* overwrite_output_dir --- True если перезаписывать каталог с результатами, иначе False
* num_train_epochs --- Количество эпох обучения
* per_device_train_batch_size ---  Размер батча для обучения
* per_device_eval_batch_size --- Размер батча при валидации
* warmup_steps --- Количество шагов  для планировщика скорости обучения
* gradient_accumulation_steps --- Для увеличения размера "виртуального" пакета

Optimizer
* lr --- Скорость обучения 
```
## Запуск
**Все скрипты необходимо запускать из корневого каталога**

