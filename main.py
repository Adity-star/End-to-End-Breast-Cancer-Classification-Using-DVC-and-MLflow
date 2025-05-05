from BCClassifier import logger 
from BCClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from BCClassifier.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from BCClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from BCClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
import multiprocessing

'''
STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
'''

'''
STAGE_NAME = "Data Preprocessing stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataPreprocessingTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
'''

'''
STAGE_NAME = "Prepare Base Model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
'''

if __name__ == '__main__':
    # Enable multiprocessing support
    multiprocessing.freeze_support()
    
    STAGE_NAME = "Model Training"
    try: 
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e