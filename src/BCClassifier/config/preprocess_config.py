import os 
from BCClassifier.constants import *
from BCClassifier.utils.common import read_yaml, create_directories
from BCClassifier.entity.config_entity import DataPreprocessingConfig
from BCClassifier.config.configuration import ConfigurationManager

def get_data_preprocessing_config() -> DataPreprocessingConfig:
    config_manager = ConfigurationManager()
    return config_manager.get_data_preprocessing_config()