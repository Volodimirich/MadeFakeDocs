import os
import gdown
import functools
import torch
from datasets import load_dataset
from transformers import TextDataset


class TypeTraining:
    TEACHER = "Learning with a teacher"
    CLM = "Causal language modeling"


def get_data(dataset_dict: dict):
    save_dir, dataset_name = dataset_dict.save_dir, dataset_dict.dataset_name
    data_path = os.path.join(save_dir, dataset_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

        link = dataset_dict.gdrive_dict[dataset_name]
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
    elif dataset_name == 'made_data':
        path_dict = {'train': os.path.join(data_path, 'filtred_df.csv'),
                     'test': None,
                     'val': os.path.join(data_path, 'filtred_df.csv')}
    elif dataset_name == 'made_valid_data':
        path_dict = {'train': os.path.join(data_path, 'val.json'),
                     'test': None,
                     'val': os.path.join(data_path, 'val.json')}
    return path_dict


def preprocess_data(examples, tokenizer, input_max_length, target_max_length, mode='part_data'):
    """
    Функция для токенизации данных где:
        * input_ids - токенизированный запрос
        * labels - токенизированный текст, соответствующий запросу

    :param examples:
    :param tokenizer:
    :param input_max_length:
    :param target_max_length:
    :param mode:
    :return:
    """
    if mode in  ['part_data', 'made_valid_data']:
        model_inputs = tokenizer(text=examples['query'],
                                 max_length=input_max_length,
                                 truncation=True, padding="max_length")
        text_arrays = [''.join(x["passage_text"]) for x in examples["passages"]]
        text_labels = [x["is_selected"] for x in examples["passages"]]
        labels = tokenizer(text_arrays, max_length=target_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs['pos_label'] = text_labels

    elif mode == "made_data":
        model_inputs = tokenizer(text=examples['query'],
                                 max_length=input_max_length,
                                 truncation=True, padding="max_length")
        labels = tokenizer(examples['body'], max_length=target_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs['pos_label'] = examples['label']

    else:
        raise NotImplementedError()

    return model_inputs


def collate_fn(batch):
    inputs_ids, passages = [], []
    #
    for item in batch:
        inputs_ids.append(item["input_ids"])
        passages.append(item["passages"])

    in_ids = torch.stack(inputs_ids)
    batch = {"input_ids": in_ids, "passages": passages}
    return batch


def groups_texts(examples, tokenizer, block_size):
    concatenated_text = {}
    for ind_example, cur_example in enumerate(examples["passages"]):
        concatenated_text[ind_example] = ""
        for ind, cur_text in enumerate(cur_example["passage_text"]):
            if cur_example["is_selected"][ind]:
                concatenated_text[ind_example] += examples["query"][ind_example]

            concatenated_text[ind_example] += cur_text

    tokenized_text = {k: tokenizer(text=concatenated_text[k], truncation=True)["input_ids"] for k in
                      concatenated_text.keys()}

    result = {
        "input_ids": []
    }
    for k, t in tokenized_text.items():
        result["input_ids"].extend(
            [t[i: i + block_size] for i in range(0, len(t) - block_size, block_size)])

    result["labels"] = result["input_ids"].copy()
    return result


def groups_texts_made(examples, tokenizer, context_size, block_size):
    """
    Функция для группировки текста и нарезка его на блоки для задачи CLM
    """

    # Склеиваем текст для последующей нарезки на блоки
    # Отдельные части разделяем eos_token токеном
    concatenated_text = {}
    for ind_example, cur_example in enumerate(examples["body"]):
        concatenated_text[ind_example] = examples["query"][ind_example]
        concatenated_text[ind_example] += " " + cur_example + tokenizer.eos_token

    tokenized_text = {k: tokenizer(text=concatenated_text[k],
                                   # truncation=True,
                                   return_overflowing_tokens=True,
                                   return_length=True,
                                   max_length=context_size,
                                   padding="max_length"
                                   )["input_ids"] for k in
                      concatenated_text.keys()}

    result = {
        "input_ids": []
    }
    for k, t in tokenized_text.items():
        result["input_ids"].extend(
            [t[i: i + block_size] for i in range(0, len(t) - block_size, block_size)])

    # Так как учим GPT на задаче CLM ей не нужны labels
    result["labels"] = result["input_ids"].copy()
    return result


def get_dataset(dataset_dict, path_list, tokenizer, total_samples=500,
                input_max_length=128, target_max_length=128,
                type_training=TypeTraining.TEACHER, type_dataset="train"):
    dataset_name, bl_size = dataset_dict.dataset_name, dataset_dict.block_size
    train_path, test_path, val_path = path_list['train'], path_list['test'], path_list['val']

    if dataset_name == 'full_data':
        train_path = path_list['train']
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=bl_size)
    elif dataset_name in {'part_data', 'made_data', "made_valid_data"}:
        INPUT_MAX_LENGTH = input_max_length
        TARGET_MAX_LENGTH = target_max_length
        NUM_PROC = 1

        fun_process_data = functools.partial(preprocess_data,
                                             tokenizer=tokenizer,
                                             input_max_length=INPUT_MAX_LENGTH,
                                             target_max_length=TARGET_MAX_LENGTH,
                                             mode=dataset_name)

        if dataset_name == 'part_data':
            dataset = load_dataset('json', data_files={'train': [train_path],
                                                       'test': [test_path],
                                                       'validation': [val_path]})
        elif dataset_name == "made_valid_data":
            dataset = load_dataset('json', data_files={'validation': [val_path]})
        else:
            dataset = load_dataset('csv', data_files={'train': [train_path], "validation": [val_path]})

        if type_dataset == "validation":
            test_dataset = dataset["validation"].select(range(min(total_samples, len(dataset["validation"]))))
            if dataset_name == 'part_data':
                test_dataset = test_dataset.map(fun_process_data, batched=True,
                                                num_proc=NUM_PROC, remove_columns=['answers',
                                                                                   "query",
                                                                                   "query_id",
                                                                                   "query_type",
                                                                                   "wellFormedAnswers"])

            elif dataset_name == "made_valid_data":
                test_dataset = test_dataset.map(fun_process_data, batched=True,
                                                num_proc=NUM_PROC, remove_columns=['query'])
            elif dataset_name == "made_data":
                test_dataset = test_dataset.map(fun_process_data, batched=True,
                                                num_proc=NUM_PROC, remove_columns=['query',
                                                                                   "url",
                                                                                   "title",
                                                                                   "meta",
                                                                                   "qlinks"])

            else:
                raise NotImplementedError()
            return test_dataset

        train_dataset = dataset['train'].select(range(min(total_samples, len(dataset['train']))))
        if type_training == TypeTraining.TEACHER:
            train_dataset = train_dataset.map(fun_process_data, batched=True, num_proc=NUM_PROC)
        elif type_training == TypeTraining.CLM:

            if dataset_name == 'part_data':
                fun_groups_texts = functools.partial(groups_texts, tokenizer=tokenizer, block_size=bl_size)
                train_dataset = train_dataset.map(fun_groups_texts, batched=True, num_proc=NUM_PROC,
                                                  remove_columns=["passages", 'answers',
                                                                  "query", "query_id",
                                                                  "query_type",
                                                                  "wellFormedAnswers"])
            else:
                fun_groups_texts = functools.partial(groups_texts_made, tokenizer=tokenizer,
                                                     block_size=bl_size, context_size=INPUT_MAX_LENGTH)
                train_dataset = train_dataset.map(fun_groups_texts, batched=True,
                                                  num_proc=NUM_PROC,
                                                  remove_columns=["label", 'query',
                                                                  "url", "title",
                                                                  "meta",
                                                                  "body",
                                                                  "qlinks"])
    else:
        raise NotImplementedError()

    return train_dataset
