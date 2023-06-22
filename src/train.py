"""
Данный модуль содержит общий пайплайн обучения для всех моделей
"""

import os

import wandb

from src.logger_config.config import LOGGING_CONFIG

import hydra
from src.modules.data import get_data, get_dataset
from src.modules.model import get_tokenizer, get_model
import torch
import logging

from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from src.modules.engine import (
    train
)
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from src.enities.training_pipeline_params import TrainingPipelineParams
from src.modules.data import TypeTraining
from torch.nn.parallel import DataParallel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def training_pipeline(params: TrainingPipelineParams):
    # wandb.login(relogin=True)
    os.environ["WANDB_ENTITY"] = "madefakedocs"
    os.environ['WANDB_PROJECT'] = params.basic.wandb_project
    logger.info(f"Name of the logging project wandb: {params.basic.wandb_project}")

    logger.info(f"Currently used device: {device}")
    dataset_path_dict = get_data(params.dataset)

    model_info = f'Pretrained {params.model.model_name}' if not params.model.use_local \
        else f'Local {params.model.model_name} from {params.model.local_path}'
    logger.info(f'Initializing the model: {model_info}')

    tokenizer = get_tokenizer(params.model.tokenizer_name)
    logger.info(f'Get tokenizer {params.model.tokenizer_name}')

    model = get_model(params.model.model_name, device, params.model.local_path, params.model.use_local)
    model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)
    logger.info('Model created')

    # Создание датасета
    train_dataset = get_dataset(params.dataset, dataset_path_dict, tokenizer,
                                total_samples=params.model.total_samples,
                                input_max_length=params.model.input_max_length,
                                target_max_length=params.model.target_max_length,
                                model_name=params.model.model_name,
                                type_training=TypeTraining.TEACHER)
    print(device)

    if params.model.model_name.lower().find("gpt") != -1:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=params.dataset.mlm)
    elif params.model.model_name.lower().find("t5") != -1:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    else:
        raise NotImplementedError(f"This model is not supported!")

    logger.info('Loader finished')
    if params.model.model_name.lower().find("gpt") != -1:
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
            report_to=None,
            save_strategy="epoch",
            fp16=True
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.train_params.lr)
    elif params.model.model_name.lower().find("t5") != -1:
        training_args = Seq2SeqTrainingArguments(
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=params.train_params.lr,
            per_device_train_batch_size=params.train_params.per_device_train_batch_size,
            optim='adafactor',
            num_train_epochs=params.train_params.num_train_epochs,
            fp16=True,
            report_to="wandb",
            output_dir=params.train_params.output_dir,
            run_name='FRED-T5-1.7B_mult_GPU',
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.train_params.lr)

    logger.info('Starting trained...')
    train(model, data_collator, train_dataset, training_args, tokenizer, optimizer, params)
    logger.info('The training is completed!')


if __name__ == "__main__":
    training_pipeline()
