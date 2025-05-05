from BCClassifier.config.configuration import ConfigurationManager
from BCClassifier.components.data_preprocessing import DataPreprocessing
from BCClassifier import logger
from BCClassifier.config.preprocess_config import get_data_preprocessing_config
from BCClassifier.utils.build_dataset import build_dataset  

STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.run_preprocessing()  


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
