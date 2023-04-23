import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import hydra
import torch
import logging
import sys
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from src.models.engine import (
    train
)
from src.enities.training_pipeline_params import TrainingPipelineParams

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def training_pipeline(params: TrainingPipelineParams):
    os.environ['WANDB_PROJECT'] = params.wandb_project
    logger.info(f"Name of the logging project wandb: {params.wandb_project}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Currently used device: {device}")

    logger.info(f'Initializing the model: {params.model_name_or_path}')
    tokenizer = GPT2Tokenizer.from_pretrained(params.tokenizer_name)

    model = GPT2LMHeadModel.from_pretrained(params.model_name_or_path).to(device)

    # Создание датасета
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=params.file_path, block_size=params.block_size)

    # Создание даталодера (нарезает текст на оптимальные по длине куски)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=params.mlm)

    training_args = TrainingArguments(
        logging_strategy="epoch",
        output_dir=params.output_dir,  # The output directory
        overwrite_output_dir=params.overwrite_output_dir,  # Overwrite the content of the output dir
        num_train_epochs=params.num_train_epochs,  # number of training epochs
        per_device_train_batch_size=params.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=params.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=params.warmup_steps,  # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=params.gradient_accumulation_steps,  # to make "virtual" batch size larger
        report_to="wandb"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    logger.info('Starting trained...')
    train(model, data_collator, train_dataset, training_args, optimizer, params)
    logger.info('The training is completed!')


if __name__ == "__main__":
    training_pipeline()

