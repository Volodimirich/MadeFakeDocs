import os
from logger_config.config import LOGGING_CONFIG

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import hydra
import torch
import logging
import gdown

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from models.engine import (
    train
)
from enities.training_pipeline_params import TrainingPipelineParams
# from ..configs.logger_config import LOGGING_CONFIG

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_data(dataset_dict: dict):
    save_dir, dataset_name = dataset_dict.save_dir, dataset_dict.dataset_name
    data_path = os.path.join(save_dir, dataset_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        
        link = dataset_dict.gdrive_dict[dataset_name]
        print(link)   
        if dataset_name == 'full_data':
            final_path = os.path.join(data_path, 'dataset.tsv.gz')
            gdown.download(link, final_path, quiet=False)
            os.system(f'gzip -d {final_path}')
        elif dataset_name == 'part_data':
            gdown.download(link['train'], os.path.join(data_path, 'train.jsonl'), quiet=False)
            gdown.download(link['test'], os.path.join(data_path, 'test.jsonl'), quiet=False)
            gdown.download(link['val'], os.path.join(data_path, 'val.jsonl'), quiet=False)

    if dataset_name == 'full_data':
        path_dict = {'train': os.path.join(data_path, 'dataset.tsv'),
                     'test': None,
                     'val': None}
    elif dataset_name == 'part_data':
        path_dict = {'train': os.path.join(data_path, 'train.jsonl'),
                     'test': os.path.join(data_path, 'test.jsonl'),
                     'val': os.path.join(data_path, 'val.jsonl')}
    return path_dict

def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'gpt2':
        return GPT2Tokenizer.from_pretrained(tokenizer_name)


def get_model(model_name, local_path='', is_local=False):
    if model_name == 'gpt2':
        model = local_path if is_local else model_name
        return GPT2LMHeadModel.from_pretrained(model).to(device)   
    
    

def get_dataset(dataset_dict, path_list, tokenizer):
    dataset_name, bl_size = dataset_dict.dataset_name, dataset_dict.block_size

    if dataset_name == 'full_data':
        train_path = path_list['train']
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, 
                            block_size=bl_size)
    elif dataset_name == 'part_data':
        train_path, test_path, val_path = path_list['train'], path_list['test'], path_list['val']
        dataset = load_dataset('json', data_files={'train': [train_path], 
                                                   'test':[test_path], 
                                                   'validation':[val_path]})
        logger.error('Functional not finished yet')
        raise NotImplementedError
    return train_dataset
        
        


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

    # GPT2Tokenizer.from_pretrained(params.tokenizer_name)

    model = get_model(params.model.model_name, params.model.local_path, params.model.use_local)
    logger.info('Model created')
    # GPT2LMHeadModel.from_pretrained(params.model_name_or_path).to(device)

    # Создание датасета
    train_dataset = get_dataset(params.dataset, dataset_path_dict, tokenizer)

    # Создание даталодера (нарезает текст на оптимальные по длине куски)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=params.dataset.mlm)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr)
    logger.info('Starting trained...')
    train(model, data_collator, train_dataset, training_args, optimizer, params)
    logger.info('The training is completed!')


if __name__ == "__main__":
    training_pipeline()

