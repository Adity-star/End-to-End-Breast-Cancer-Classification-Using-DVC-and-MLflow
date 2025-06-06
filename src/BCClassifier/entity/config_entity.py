from dataclasses import dataclass
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str 
    local_data_file: Path 
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    base_path: Path
    train_split: float
    val_split: float
    training_dir: Path
    validation_dir: Path
    testing_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    update_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_classes: int
    params_weights: str


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_model_path: Path
    training_data: Path
    validation_data: Path
    test_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_classes: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_to_model: Path
    training_data: Path
    base_model_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int   
    params_weights: str
    params_include_top: bool

