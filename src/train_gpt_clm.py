"""
Данный модуль содержит пайплайн обучения GPT с помощью TextDataset
"""
import os

import wandb

from logger_config.config import LOGGING_CONFIG

import hydra
from modules.data import get_data, get_dataset
from modules.model import get_tokenizer, get_model
import torch
import logging

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from modules.engine import (
    train
)
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback)

from enities.training_pipeline_params import TrainingPipelineParams
from modules.data import TypeTraining

# from ..configs.logger_config import LOGGING_CONFIG

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()

def get_made_data_gpt(dataset_dict, tokenizer):
    from datasets import load_dataset
    """ Функция сохраняет """
    save_dir, dataset_name = dataset_dict.save_dir, dataset_dict.dataset_name
    data_path = os.path.join(save_dir, dataset_name)

    path_dict = {'train': os.path.join(data_path, 'filtred_df.csv'),
                 'test': None,
                 'val': os.path.join(data_path, 'filtred_df.csv')}

    train_path, test_path, val_path = path_dict['train'], path_dict['test'], path_dict['val']
    dataset = load_dataset('csv', data_files={'train': [train_path], "validation": [val_path]})
    name_folder = os.path.dirname(os.path.abspath(dataset_dict.path_to_save_txt))
    os.makedirs(name_folder, exist_ok=True)
    dataset = dataset["train"]
    with open(dataset_dict.path_to_save_txt, "w", encoding="utf-8") as file:
        for text in dataset:
            file.write(text["query"] + " " + tokenizer.sep_token + " " + text["body"] + " " + tokenizer.eos_token)
    return dataset_dict.path_to_save_txt

def get_part_data_gpt(dataset_dict, tokenizer):
    from datasets import load_dataset
    """ Функция сохраняет """
    save_dir, dataset_name = dataset_dict.save_dir, dataset_dict.dataset_name
    data_path = os.path.join(save_dir, dataset_name)

    path_dict = {'train': os.path.join(data_path, 'train.jsonl'),
                 'test': None,
                 'val': os.path.join(data_path, 'val.jsonl')}

    train_path, test_path, val_path = path_dict['train'], path_dict['test'], path_dict['val']
    dataset = load_dataset('json', data_files={'train': [train_path], "validation": [val_path]})
    name_folder = os.path.dirname(os.path.abspath(dataset_dict.path_to_save_txt))
    os.makedirs(name_folder, exist_ok=True)
    dataset = dataset["train"]
    with open(dataset_dict.path_to_save_txt, "w", encoding="utf-8") as file:
        for example in dataset:
            for ind in range(len(example["passages"]["is_selected"])):
                if not example["passages"]["is_selected"][ind]:
                    continue

                file.write(example["query"] + " " + example["passages"]["passage_text"][ind] + " " + tokenizer.eos_token)
    return dataset_dict.path_to_save_txt


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def training_pipeline(params: TrainingPipelineParams):
    # wandb.login(relogin=True)
    os.environ["WANDB_ENTITY"] = "madefakedocs"
    os.environ['WANDB_PROJECT'] = params.basic.wandb_project
    logger.info(f"Name of the logging project wandb: {params.basic.wandb_project}")

    logger.info(f"Currently used device: {device}")


    model_info = f'Pretrained {params.model.model_name}' if not params.model.use_local \
        else f'Local {params.model.model_name} from {params.model.local_path}'
    logger.info(f'Initializing the model: {model_info}')

    tokenizer = get_tokenizer(params.model.tokenizer_name)
    logger.info(f'Get tokenizer {params.model.tokenizer_name}')
    if params.dataset.dataset_name == "made_data":
        dataset_path = get_made_data_gpt(params.dataset, tokenizer)
    elif params.dataset.dataset_name == "part_data":
        dataset_path = get_part_data_gpt(params.dataset, tokenizer)
    else:
        raise NotImplementedError("This dataset is not supported!")

    train_dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_path, block_size=params.dataset.block_size)
    model = get_model(params.model.model_name, device, params.model.local_path, params.model.use_local)
    # New
    model.resize_token_embeddings(len(tokenizer))

    logger.info('Model created')

    print(device)
    # Создание даталодера (нарезает текст на оптимальные по длине куски)
    # TODO Решить, нужен ли нам collator, выбрать оптимальную подгрузку данных
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=params.dataset.mlm)
    # data_collator = DataCollatorForSeq2Seq(tokenizer)
    logger.info('Loader finished')
    training_args = TrainingArguments(
        logging_strategy="epoch",
        output_dir=params.train_params.output_dir,  # The output directory
        overwrite_output_dir=params.train_params.overwrite_output_dir,  # Overwrite the content of the output dir
        num_train_epochs=params.train_params.num_train_epochs,  # number of training epochs
        per_device_train_batch_size=params.train_params.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=params.train_params.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=params.train_params.warmup_steps,  # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=params.train_params.gradient_accumulation_steps,
        # to make "virtual" batch size larger
        report_to="wandb",
        # save_strategy="epoch",
        fp16=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.train_params.lr)
    logger.info('Starting trained...')
    train(model, data_collator, train_dataset, training_args, tokenizer, optimizer, params)
    logger.info('The training is completed!')


if __name__ == "__main__":
    training_pipeline()

