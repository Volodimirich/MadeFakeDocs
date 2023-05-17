import os
import gdown

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
    return path_dict


def get_dataset(dataset_dict, path_list, tokenizer, total_samples=500,
                input_max_length=32, target_max_length=128,
                type_training=TypeTraining.TEACHER, type_dataset="train"):
    dataset_name, bl_size = dataset_dict.dataset_name, dataset_dict.block_size
    train_path, test_path, val_path = path_list['train'], path_list['test'], path_list['val']

    if dataset_name == 'full_data':
        train_path = path_list['train']
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path,
                                    block_size=bl_size)
    if dataset_name == 'part_data':
        INPUT_MAX_LENGTH = input_max_length
        TARGET_MAX_LENGTH = target_max_length
        print(INPUT_MAX_LENGTH, TARGET_MAX_LENGTH, total_samples)

        def preprocess_data(examples):
            model_inputs = tokenizer(text=examples['query'],
                                     max_length=INPUT_MAX_LENGTH,
                                     truncation=True, padding="max_length")
            text_arrays = [''.join(x["passage_text"]) for x in examples["passages"]]
            text_labels = [x["is_selected"] for x in examples["passages"]]
            labels = tokenizer(text_arrays, max_length=TARGET_MAX_LENGTH, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            model_inputs['pos_label'] = text_labels
            return model_inputs

        # TODO: Загружать исключительно нужную часть
        dataset = load_dataset('json', data_files={'train': [train_path],
                                                   'test': [test_path],
                                                   'validation': [val_path]})

        train_dataset = dataset['train'].select(range(total_samples))
        if type_dataset == "validation":
            test_dataset = dataset["validation"].select(range(total_samples))
            test_dataset = test_dataset.map(preprocess_data, batched=True,
                                            load_from_cache_file=False,
                                            num_proc=1, remove_columns=['answers',
                                                                        "query",
                                                                        "query_id",
                                                                        "query_type",
                                                                        "wellFormedAnswers"])
            def padding_passages(examples):
                max_length = 10

                for cnt in range(max_length - len(examples["passages"]["passage_text"])):
                    examples["passages"]["passage_text"].append('')
                    examples["passages"]["is_selected"].append(0)
                    examples["passages"]["url"].append("")
                return examples

            test_dataset = test_dataset.map(padding_passages)
            return test_dataset

        if type_training == TypeTraining.TEACHER:
            train_dataset = train_dataset.map(preprocess_data, batched=True,
                                              num_proc=32)
        elif type_training == TypeTraining.CLM:
            def group_texts(examples):
                # Concatenate all texts.
                block_size = 64
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

            train_dataset = train_dataset.map(group_texts, batched=True, num_proc=32,
                                              remove_columns=["passages", 'answers',
                                                              "query", "query_id",
                                                              "query_type",
                                                              "wellFormedAnswers"])

    return train_dataset

#
