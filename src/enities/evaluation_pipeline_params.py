from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class EvaluationPipelineParams:
    # Logging metrics
   # wandb_project: str

    # Model
    model_name_or_path: str
    tokenizer_name: str

    # Dataset
    dataset_name: str
    dataset_version: str

    # EvaluationArguments
    output_dir: str
    batch_size: int
    num_sample_file: int # Количество строк во временном файле оценки



EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluating_pipeline_params(path: str) -> EvaluationPipelineParams:
    with open(path, "r") as input_stream:
        schema = EvaluationPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
