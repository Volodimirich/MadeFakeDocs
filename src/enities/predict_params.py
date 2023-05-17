from dataclasses import dataclass


@dataclass()
class PredictingParams:
    do_sample: bool
    num_beams: int
    temperature: float
    top_p: float
    max_length: int
