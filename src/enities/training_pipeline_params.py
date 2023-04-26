from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    # Logging metrics
    wandb_project: str

    # Model
    model_name_or_path: str
    tokenizer_name: str
    path_save_model: str

    # Dataset
    file_path: str
    block_size: int
    mlm: bool

    # TrainingArguments
    pre_trained: bool
    output_dir: str
    overwrite_output_dir: bool
    num_train_epochs: int  # number of training epochs
    per_device_train_batch_size: int  # batch size for training
    per_device_eval_batch_size: int  # batch size for evaluation
    warmup_steps: int  # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps: int  # to make "virtual" batch size larger

    # Optimizer
    lr: float


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
