from model_functions import make_predictions, get_results
from loguru import logger


predictions = make_predictions()

accuracy, classification_report = get_results(predictions)

logger.info("Finished making predictions, accuracy = {}".format(accuracy))
