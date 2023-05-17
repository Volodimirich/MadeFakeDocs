import os
import sys
import hydra
import torch
import shutil
import logging
# import pandas as pd
from tqdm import tqdm
# from pathlib import Path
# from datetime import datetime
# from datasets import load_dataset
from modules.data import get_data, get_dataset
from modules.model import get_tokenizer, get_model
from logger_config.config import LOGGING_CONFIG
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.enities.evaluation_pipeline_params import EvaluationPipelineParams
# from src.metrics.ranking_metrics import RankingMetrics, LaBSE, Bm25
from modules.engine import predict
from modules.data import TypeTraining
from docs_ranking_metrics import LaBSE, Bm25, RankingMetrics

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='evaluate_config')
def evaluate(params: EvaluationPipelineParams):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Currently used device: {device}")
    dataset_path_dict = get_data(params.dataset)

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
                               type_dataset="valid")

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained(params.model_name_or_path).to(device)
    # logger.info(f'Initializing the model: {params.model_name_or_path}')
    #
    # tmp_folder = "_tmp_"
    # # Путь до временного каталога с предиктами модельки
    # path_to_tmp_folder = os.path.join(params.output_dir, tmp_folder)
    # if os.path.exists(path_to_tmp_folder):
    #     shutil.rmtree(path_to_tmp_folder)
    #
    # os.makedirs(path_to_tmp_folder)
    # logger.info(f'Uploading a dataset: {params.dataset_name}')
    # dataset = load_dataset(params.dataset_name, params.dataset_version)

    metrics = [LaBSE(), Bm25()]
    rank_metrics = RankingMetrics(metrics)
    #
    # batch_queries = []
    # queries, queries_ids = [], []
    # generated_texts, is_selected = [], []
    logger.info(f'The dataset is loaded!')
    for row in tqdm(test_dataset,
                    total=len(test_dataset),
                    desc="Generating predictions...",
                    ncols=80):
        query = row["query"]
        labels = row["passages"]["is_selected"]
        sentences = row["passages"]["passage_text"]

        generated_text = predict([query], model, tokenizer, device, params.model.predict_param)
        # print(f"Query: {query}\n Generated Text: {generated_text[0]}")
        labels.append(RankingMetrics.FAKE_DOC_LABEL)

        sentences.append(generated_text[0][len(query):])
        rank_metrics.update(query, sentences, labels)

    #     if len(batch_queries) == params.batch_size:
    #         generated_text = predict(batch_queries, model, tokenizer, device)
    #         generated_texts.extend([x.replace('\n', '') for x in generated_text])
    #         queries.extend(batch_queries)
    #         batch_queries.clear()
    #
    #     if len(generated_texts) > params.num_sample_file:
    #         pd.DataFrame({"queries_ids": queries_ids,
    #                       "queries": queries,
    #                       "generated_texts": generated_texts
    #                       }).to_csv(os.path.join(path_to_tmp_folder,
    #                                              str(datetime.now().timestamp()),
    #                                              ".csv"), index=False)
    #         generated_texts.clear()
    #         queries_ids.clear()
    #         queries.clear()
    #
    #     if len(batch_queries) < params.batch_size:
    #         batch_queries.append(query)
    #         queries_ids.append(row["query_id"])
    #
    # if len(batch_queries):
    #     generated_texts.extend(predict(batch_queries, model, tokenizer, device))
    #     queries.extend(batch_queries)
    #     batch_queries.clear()
    #
    # if len(generated_texts):
    #     pd.DataFrame({"queries_ids": queries_ids,
    #                   "queries": queries,
    #                   "generated_texts": generated_texts
    #                   }).to_csv(os.path.join(path_to_tmp_folder,
    #                                          str(datetime.now().total_seconds)) + ".csv")
    #
    # logger.info("Evaluating started!")
    # list_files = [float(Path(x).stem) for x in os.listdir(path_to_tmp_folder)]
    # list_files.sort()
    # metrics = [LaBSE()]
    # rank_metrics = RankingMetrics(metrics)
    # for file_name in list_files:
    #     pass
    print(f"Results:")
    for k, v in rank_metrics.get().items():
        print(f"{k}: {v}")
    logger.info("Evaluating done!")


if __name__ == "__main__":
    evaluate()
