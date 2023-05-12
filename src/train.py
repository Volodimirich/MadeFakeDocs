import os
from logger_config.config import LOGGING_CONFIG

import hydra
from modules.data import get_data, get_dataset
from modules.model import get_tokenizer, get_model
import torch
import logging

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from modules.engine import (
    train
)
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel, 
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback)

from enities.training_pipeline_params import TrainingPipelineParams
# from ..configs.logger_config import LOGGING_CONFIG

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def training_pipeline(params: TrainingPipelineParams):
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
    logger.info('Model created')

    # Создание датасета
    train_dataset = get_dataset(params.dataset, dataset_path_dict, tokenizer, 
                                total_samples=params.model.total_samples, 
                                input_max_length=params.model.input_max_length,
                                target_max_length=params.model.target_max_length)
    print(device)
    # Создание даталодера (нарезает текст на оптимальные по длине куски)
    # TODO Решить, нужен ли нам collator, выбрать оптимальную подгрузку данных
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    # mlm=params.dataset.mlm)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    logger.info('Loader finished')
    training_args = TrainingArguments(
        logging_strategy="epoch",
        output_dir=params.train_params.output_dir,  # The output directory
        overwrite_output_dir=params.train_params.overwrite_output_dir,  # Overwrite the content of the output dir
        num_train_epochs=params.train_params.num_train_epochs,  # number of training epochs
        per_device_train_batch_size=params.train_params.per_device_train_batch_size,  # batch size for training
        per_device_eval_batch_size=params.train_params.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=params.train_params.warmup_steps,  # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=params.train_params.gradient_accumulation_steps,  # to make "virtual" batch size larger
        report_to="wandb"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.train_params.lr)
    logger.info('Starting trained...')
    train(model, data_collator, train_dataset, training_args, optimizer, params)
    logger.info('The training is completed!')


if __name__ == "__main__":
    training_pipeline()

