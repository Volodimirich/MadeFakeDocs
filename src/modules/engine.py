import torch
from transformers import Trainer, TrainingArguments


def evaluate(model, tokenizer, device):
    pass


def train(model, data_collator, train_dataset, training_args, optimizer, params):
    '''
    Функция для обучения

    Parameters
    ------------
    model: ``
        Модель для обучения
    data_collator: ``
    train_dataset: ``
    training_args: ``
    optimizer: ``
        Оптимизатор
    training_config: ``

    Returns
    ------------
    `list`
        Результаты тренировки модели
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None)
    )

    training_results = trainer.train()
    # trainer.save_model(params.path_save_model)
    trainer.save_model()
    return training_results


def predict(queries, model, tokenizer, device, predict_config):
    '''
    Функция для генерации текста по входным запросам пользователя

    Parameters
    ------------
    queries: `list`
        Список запросов для генерации ответов
    model: ``
        Модель для генерации текста
    tokenizer: ``
        Токенизатор для разделения текста на токены
    predict_config: ``
        Конфигурационный файл
    device: ``
        Устройство, на котором будут производиться вычисления

    Returns
    ------------
    `list`
        Список сгенерированных текстов для каждого входного запроса
    '''
    results = []
    model.eval()
    with torch.no_grad():
        for query in queries:
            input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
            out = model.generate(input_ids,
                                 do_sample=True,
                                 num_beams=2,
                                 temperature=1.5,
                                 top_p=0.9,
                                 max_length=100,
                                 )

            generated_text = list(map(tokenizer.decode, out))[0]
            results.append(generated_text)
    return results
