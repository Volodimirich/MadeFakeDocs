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
from modules.data import get_data, get_dataset
from modules.model import get_tokenizer, get_model
from logger_config.config import LOGGING_CONFIG
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.enities.evaluation_pipeline_params import EvaluationPipelineParams
from src.modules.data import collate_fn
from transformers import DataCollatorForSeq2Seq
# from src.metrics.ranking_metrics import RankingMetrics, LaBSE, Bm25
from modules.engine import predict
from modules.data import TypeTraining
from torch.utils.data import DataLoader
from docs_ranking_metrics import LaBSE, Bm25, RankingMetrics, USE, MsMarcoST,MsMarcoCE
import datetime

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='evaluate_config')
def evaluate(params: EvaluationPipelineParams):
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
    # TODO: СОхранить запускаемые параметры

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
                               type_dataset="validation")

    test_dataset.set_format(type="torch", columns=["input_ids", "passages"])

    eval_dataloader = DataLoader(test_dataset, batch_size=params.model.batch_size, collate_fn=collate_fn)
    number_batches_save = [random.randint(0, len(eval_dataloader) - 1) for _ in range(params.result.number_examples_batch)]
    examples_of_generation = []

    metrics = [LaBSE(), Bm25()]
    rank_metrics = RankingMetrics(metrics, [2, 3, 4])

    logger.info(f'The dataset is loaded!')
    for ind_batch, row in tqdm(enumerate(eval_dataloader),
                               total=len(eval_dataloader),
                               desc="Generating predictions...",
                               ncols=80):

        query = tokenizer.batch_decode(row["input_ids"], skip_special_tokens=True)

        labels = [row["passages"][ind]["is_selected"].detach().cpu().tolist() for ind in range(len(row["passages"]))]
        sentences = [row["passages"][ind]["passage_text"] for ind in range(len(row["passages"]))]

        assert len(query) <= params.model.batch_size, "Неверный формат запросов!"
        assert len(query) == len(labels), "Размеры должны совпадать! Первое измерение BATCH_SIZE"
        assert len(query) == len(sentences), "Размеры должны совпадать! Первое измерение BATCH_SIZE"

        generated_text = predict(row["input_ids"], model, tokenizer, device, params.model.predict_param)

        for ind in range(len(query)):
            updated_sequence = list(sentences[ind])
            if params.result.gpt_postprocessing and params.model.model_name.find("gpt"):
                result_text = generated_text[ind][len(query[ind]):]
            else:
                result_text = generated_text[ind]

            updated_sequence.append(result_text)

            updated_labels = list(labels[ind])
            updated_labels.append(RankingMetrics.FAKE_DOC_LABEL)

            rank_metrics.update(query[ind], updated_sequence, updated_labels)

            if params.result.save_examples and ind_batch in number_batches_save:
                examples = {
                    "query": query[ind],
                    "generated_text": result_text
                }
                examples_of_generation.append(examples)

    logger.info(f"Results:")
    result_metrics = rank_metrics.get()
    for k, v in result_metrics.items():
        logger.info(f"{k}: {v}")

    logger.info("Evaluating done!")
    path_to_save_metrics = os.path.join(folder_result_name, "Metrics.json")
    logger.info(f"Save metrics: {path_to_save_metrics}")
    with open(path_to_save_metrics, 'w') as f:
        json.dump(result_metrics, f)

    if params.result.save_examples:
        path_to_save_examples = os.path.join(folder_result_name, "Examples.json")
        logger.info(f"Save examples: {path_to_save_examples}")
        with open(path_to_save_examples, 'w', encoding="utf-8") as f:
            json.dump(examples_of_generation, f, ensure_ascii=False)

if __name__ == "__main__":
    evaluate()
