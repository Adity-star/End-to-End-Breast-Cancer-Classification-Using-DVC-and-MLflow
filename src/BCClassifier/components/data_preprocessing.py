from BCClassifier.entity.config_entity import DataPreprocessingConfig
from BCClassifier.utils.build_dataset import build_dataset 
from BCClassifier import logger

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def run_preprocessing(self):
        try:
            logger.info("Starting image preprocessing and dataset split...")
            build_dataset(
                orig_input_dataset=self.config.root_dir,
                train_path=self.config.training_dir,
                val_path=self.config.validation_dir,
                test_path=self.config.testing_dir,
                train_split=self.config.train_split,
                val_split=self.config.val_split
            )
            logger.info("Preprocessing and dataset split completed.")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise e
