# Описать обучения модели ruGPT3 и эксперименты на новых данных и ранжировщике

Конфиг обучения:
```commandline
# Logging metrics
basic:
  wandb_project: "MadeFakeDocs"

# Model
model:
  model_name: "ai-forever/rugpt3large_based_on_gpt2"
  tokenizer_name: "ai-forever/rugpt3large_based_on_gpt2"
  local_path: './result_gpt3_new_data_train'
  use_local: False
  path_save_model: str
  input_max_length: 256
  target_max_length: 1024
  total_samples: 2000
  type_of_training: "clm"


# Dataset
dataset:
  save_dir: 'data'
  dataset_name: "made_data"
  block_size: 256
  mlm: False
  path_to_save_txt: "./data/gpt_training/txt_training.txt"
  name_txt_file: "txt_training.txt"


# TrainingArguments
train_params:
  pre_trained: bool
  output_dir: "./result_gpt3_new_data_train"
  overwrite_output_dir: True
  num_train_epochs: 10  # number of training epochs
  per_device_train_batch_size: 1  # batch size for training
  per_device_eval_batch_size: 1  # batch size for evaluation
  warmup_steps: 10  # number of warmup steps for learning rate scheduler
  gradient_accumulation_steps: 16   # to make "virtual" batch size larger

  # Optimizer
  lr: 1e-5

```