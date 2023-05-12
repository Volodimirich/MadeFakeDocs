import os
import gdown

from datasets import load_dataset
from transformers import TextDataset

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
                input_max_length=32, target_max_length=128):
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
            model_inputs = tokenizer(text=examples['query'], max_length=INPUT_MAX_LENGTH, truncation=True)
            text_arrays = [''.join(x["passage_text"]) for x in examples["passages"]] 
            text_labels = [x["is_selected"] for x in examples["passages"]]
            labels = tokenizer(text_arrays, max_length=TARGET_MAX_LENGTH, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            model_inputs['pos_label'] = text_labels
            return model_inputs
        
        dataset = load_dataset('json', data_files={'train': [train_path], 
                                                   'test':[test_path], 
                                                   'validation':[val_path]})
        
       
        train_dataset = dataset['train'].select(range(total_samples))
        train_dataset = train_dataset.map(preprocess_data, batched=True, num_proc=128)
    return train_dataset
        
# 