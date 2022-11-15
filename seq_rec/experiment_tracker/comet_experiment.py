import os
from dotenv import load_dotenv
load_dotenv()

from comet_ml import Experiment


class CometExperiment(Experiment):
    def __init__(self, **kwargs):
        import comet_ml
        comet_ml.init(api_key=os.environ['COMET_API_KEY'], project_name='seq-rec')
        super().__init__(**kwargs)
