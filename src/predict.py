import os
import sys
import hydra
import numpy as np
import torch
import json
import shutil
import logging
# import pandas as pd
from tqdm import tqdm
import random
import yaml
# from pathlib import Path
# from datetime import datetime
# from datasets import load_dataset
from modules.data import get_data, get_dataset
from modules.model import get_tokenizer, get_model
from logger_config.config import LOGGING_CONFIG
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.enities.evaluation_pipeline_params import EvaluationPipelineParams
from transformers import DataCollatorForSeq2Seq
# from src.metrics.ranking_metrics import RankingMetrics, LaBSE, Bm25
from modules.engine import predict
from modules.data import TypeTraining
from torch.utils.data import DataLoader
from docs_ranking_metrics import LaBSE, Bm25, RankingMetrics
import datetime

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='predict_config')
def predict_pipeline(params: EvaluationPipelineParams):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    folder_result_name = params.result.launch_description + '_' + \
                         str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M"))

    folder_result_name = os.path.join(params.result.path_to_save_result, folder_result_name)
    os.makedirs(params.result.path_to_save_result, exist_ok=True)
    os.makedirs(folder_result_name, exist_ok=True)
    path_to_save_params = os.path.join(folder_result_name, "params.yaml")
    try:
        shutil.copyfile(os.path.join('./configs', 'evaluate_config.yaml'), path_to_save_params)
    except FileNotFoundError:
        logger.warning("The configuration file could not be copied!")

    logger.info(f"Currently used device: {device}")
    dataset_path_dict = get_data(params.dataset)
    logger.info(f"The folder where the result is saved: {folder_result_name}")

    model_info = f'Pretrained {params.model.model_name}' if not params.model.use_local \
        else f'Local {params.model.model_name} from {params.model.local_path}'
    logger.info(f'Initializing the model: {model_info}')

    tokenizer = get_tokenizer(params.model.tokenizer_name)
    logger.info(f'Get tokenizer {params.model.tokenizer_name}')

    model = get_model(params.model.model_name, device, params.model.local_path, params.model.use_local)
    model.resize_token_embeddings(len(tokenizer))
    # Создание датасета
    test_dataset = get_dataset(params.dataset, dataset_path_dict, tokenizer,
                               total_samples=params.model.total_samples,
                               input_max_length=params.model.input_max_length,
                               target_max_length=params.model.target_max_length,
                               model_name=params.model.model_name,
                               type_training=TypeTraining.CLM,
                               type_dataset="validation")

    test_dataset.set_format(type="torch", columns=["input_ids"])
    eval_dataloader = DataLoader(test_dataset, batch_size=params.model.batch_size)
    examples_of_generation = []

    logger.info(f'The dataset is loaded!')
    for ind_batch, row in tqdm(enumerate(eval_dataloader),
                               total=len(eval_dataloader),
                               desc="Generating predictions...",
                               ncols=80):

        query = tokenizer.batch_decode(row["input_ids"], skip_special_tokens=True)
        generated_text = predict(row["input_ids"], model, tokenizer, device, params.model.predict_param)

        for ind in range(len(query)):
            result_text = generated_text[ind]

            examples = {
                "query": query[ind],
                "generated_text": result_text
            }
            examples_of_generation.append(examples)

    path_to_save_examples = os.path.join(folder_result_name, "Examples.json")
    logger.info(f"Save examples: {path_to_save_examples}")
    with open(path_to_save_examples, 'w', encoding="utf-8") as f:
        json.dump(examples_of_generation, f, ensure_ascii=False)


if __name__ == "__main__":
    predict_pipeline()
