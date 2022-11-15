import logging
logger = logging.getLogger(__name__)

from contextlib import contextmanager


class MockExperiment:
    """ This class has the same interface as Comet Experiment
    It's created so that we can run MockExperiment quickly without changing much the code
    """
    def __init__(self, **kwargs):
        """ Mock initializing

        Args:
            kwargs: keyword arguments that get passed are not processed. This is to
                match the interface of Comet Experiment
        """
        logger.info("Starting mock experiment tracker...")

    @contextmanager
    def train(self):
        """ Mock experiment does not need to implement anything
        """
        yield

    @contextmanager
    def test(self):
        """ Mock experiment does not need to implement anything
        """
        yield

    def log_parameters(self, params: dict):
        """ Print to console

        Args:
            params (dict): model hyper parameters
        """
        print(params)

    def log_metrics(self, metrics: dict):
        """ Print to console

        Args:
            metrics (dict): result of evaluation metrics
        """
        print(metrics)

    def end(self):
        """ Mock experiment does not need to implement anything
        """
        pass
