"""
Данный модель содержит пайплайн генерации ответов на вопросы для ряда моделей
"""

import os
import hydra
import torch
import json
import shutil
import logging
from modules.model import get_tokenizer, get_model
from logger_config.config import LOGGING_CONFIG
from src.enities.evaluation_pipeline_params import EvaluationPipelineParams
import glob

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

    logger.info(f"The folder where the result is saved: {folder_result_name}")

    model_info = f'Pretrained {params.model.model_name}' if not params.model.use_local \
        else f'Local {params.model.model_name} from {params.model.local_path}'
    logger.info(f'Initializing the model: {model_info}')

    tokenizer = get_tokenizer(params.model.tokenizer_name)
    logger.info(f'Get tokenizer {params.model.tokenizer_name}')
    # Список вопросов для генерации ответов на них
    texts = ["антенна для цт вязьма", "аромомасла снижение аппетита",
             "архивная выписка из домой книги через госуслуги"]

    # В данном случае params.model.local_path это путь до дирректории с checkpoints модели
    for path_to_model in glob.glob(params.model.local_path + '//checkpoint-*'):
        print(f"Cur path: {path_to_model}")
        params.model.local_path = path_to_model
        model = get_model(params.model.model_name, device, params.model.local_path, params.model.use_local)
        model.resize_token_embeddings(len(tokenizer))
        results = []
        for query in texts:
            encoded_input = tokenizer.encode(query, return_tensors='pt').to(device)
            out = model.generate(encoded_input,
                                 pad_token_id=tokenizer.eos_token_id,
                                 do_sample=params.model.predict_param.do_sample,
                                 num_beams=params.model.predict_param.num_beams,
                                 temperature=params.model.predict_param.temperature,
                                 top_p=params.model.predict_param.top_p,
                                 max_length=params.model.predict_param.max_length,
                                 )

            generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)
            examples = {
                "query": query,
                "generated_text": generated_text[0]
            }

            results.append(examples)

        path_to_model = os.path.basename(os.path.normpath(path_to_model))
        path_to_save = os.path.join(folder_result_name, path_to_model)

        os.makedirs(path_to_save, exist_ok=True)
        path_to_save = os.path.join(path_to_save, "results.json")
        logger.info(f"Save examples: {path_to_save}")
        with open(path_to_save, 'w', encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    predict_pipeline()
