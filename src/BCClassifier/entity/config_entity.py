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
