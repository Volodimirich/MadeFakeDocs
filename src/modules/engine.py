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


def predict(batch_queries, model, tokenizer, device, predict_config):
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
    model.to(device)
    batch_queries = batch_queries.to(device)
    with torch.no_grad():
        out = model.generate(batch_queries,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=predict_config.do_sample,
                             num_beams=predict_config.num_beams,
                             temperature=predict_config.temperature,
                             top_p=predict_config.top_p,
                             max_length=predict_config.max_length,
                             )

        generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(generated_text) #[text.detach().cpu().tolist() for text in generated_text])
    return results
