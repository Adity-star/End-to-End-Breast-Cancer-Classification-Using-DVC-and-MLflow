from BCClassifier.config.configuration import ConfigurationManager
from BCClassifier.components.model_evaluation_mlflow import Evaluation
from BCClassifier import logger 


STAGE_NAME = "Model Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()