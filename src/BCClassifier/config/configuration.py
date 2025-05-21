import os 
from BCClassifier.constants import *
from BCClassifier.utils.common import read_yaml,create_directories,save_json
from BCClassifier.entity.config_entity import (DataIngestionConfig, 
                                              DataPreprocessingConfig, 
                                              PrepareBaseModelConfig,
                                              TrainingConfig,
                                              EvaluationConfig,
                                              CancernetConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.base_model_path = os.path.join(self.config.prepare_base_model.root_dir, self.config.prepare_base_model.base_model_path)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        params = self.params.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            base_path=config.base_path,
            train_split=params.train_split,
            val_split=params.val_split,
            training_dir=config.training_dir,
            validation_dir=config.validation_dir,
            testing_dir=config.testing_dir
        )

        return data_preprocessing_config     

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            update_base_model_path=Path(config.updated_base_model_path), 
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHT,  
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_preprocessing.training_dir)
        validation_data = os.path.join(self.config.data_preprocessing.validation_dir)
        test_data = os.path.join(self.config.data_preprocessing.testing_dir)
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir = Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            updated_model_path = Path(training.updated_model_path),
            training_data = Path(training_data),
            validation_data = Path(validation_data),
            test_data = Path(test_data),
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE,
            params_learning_rate = params.LEARNING_RATE,
            params_include_top = params.INCLUDE_TOP,
            params_classes = params.CLASSES
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_to_model="artifacts/training/model.h5",
            training_data = "artifacts/dataset/idc/training",
            mlflow_uri = "https://dagshub.com/Adity-star/End-to-End-Breast-Cancer-Classification-Using-DVC-and-MLflow.mlflow",
            base_model_path=Path(self.base_model_path),
            all_params = self.params,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_weights=self.params.WEIGHT,
            params_include_top = self.params.INCLUDE_TOP
        )

        return eval_config
    
   
