from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from src.enities.predict_params import PredictingParams


@dataclass()
class EvaluationPipelineParams:
    # Model
    model_name: str
    tokenizer_name: str
    local_path: str
    use_local: bool
    path_save_model: str
    total_samples: str
    predict_param: PredictingParams

    # Dataset
    dataset_name: str
    dataset_version: str

    # EvaluationArguments
    output_dir: str
    batch_size: int
    num_sample_file: int  # Количество строк во временном файле оценки


EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluating_pipeline_params(path: str) -> EvaluationPipelineParams:
    with open(path, "r") as input_stream:
        schema = EvaluationPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
