import os
import gdown

from datasets import load_dataset
from transformers import TextDataset, DataCollatorForLanguageModeling

def preprocess_data(tokenizer, examples, input_column, 
                    target_column, INPUT_MAX_LENGTH = 32, TARGET_MAX_LENGTH=512):
    model_inputs = tokenizer(text=examples[input_column], max_length=INPUT_MAX_LENGTH, truncation=True)

    labels = tokenizer(examples[target_column], max_length=TARGET_MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


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


def get_dataset(dataset_dict, path_list, tokenizer, train_size=0.85, total_samples=50000, seed=42):
    dataset_name, bl_size = dataset_dict.dataset_name, dataset_dict.block_size
    train_path, test_path, val_path = path_list['train'], path_list['test'], path_list['val']

    if dataset_name == 'full_data':
        # input_column, target_column = 'query', 'text'

        # TODO Выбрать единый способ подгрузки данных.
        # train_dataset = load_dataset('csv', data_files=train_path, 
                    #    delimiter='\t', column_names=["url", input_column, target_column])
        
        
        # train_test_dataset = (dataset['train'].select(range(TOTAL_SAMPLES))
        #               .filter(lambda example: isinstance(example[INPUT_COLUMN], str) and isinstance(example[TARGET_COLUMN], str))
        #               .train_test_split(shuffle=True, train_size=TRAIN_SIZE, seed=SEED)
        # )
        
        # train_test_dataset.map(preprocess_data, batched=True, num_proc=128, remove_columns=["url", INPUT_COLUMN, TARGET_COLUMN]).save_to_disk('/content/drive/MyDrive/made_fake_documents/T5_tokenized_dataset')

        # 
        train_path = path_list['train']
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, 
                            block_size=bl_size)
    if dataset_name == 'part_data':
        INPUT_MAX_LENGTH = 32
        TARGET_MAX_LENGTH = 128
        
        def preprocess_data(examples):
            model_inputs = tokenizer(text=examples['query'], max_length=INPUT_MAX_LENGTH, truncation=True)
            text_arrays = [''.join(x["passage_text"]) for x in examples["passages"]] 
            labels = tokenizer(text_arrays, max_length=TARGET_MAX_LENGTH, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = load_dataset('json', data_files={'train': [train_path], 
                                                   'test':[test_path], 
                                                   'validation':[val_path]})
        
        
       
        train_dataset = dataset['train'].select(range(total_samples))
        train_dataset = train_dataset.map(preprocess_data, batched=True, num_proc=128)
    return train_dataset
        
# 