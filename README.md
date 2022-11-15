# Sequential Recommendation
This project is inspired by Tensorflow Recommenders. It is a demo of building a recommender model with a retrieval stage (not including ranking stage) that can provides recommendations of merchant based on the current session activities including the merchants a user have viewed and his search terms.

We use GCP as our infras for both training and serving.

# Components
- Notebooks that contain development in which experiments are tracked using Comet AI and model analysis to identify areas of improvement for later iterations
- Example model architecture is depicted in `model_architecture.md`
- Deployment of model artifact to GCP
- Pipeline written using Kubeflow to submit training job to GCP Vertex AI Pipelines. We have two kinds of pipeline here: one is for full retraining and one is for incremental retraining to save resources updating a model
- We use Vertex AI Serving for management of model endpoints after training

# How to use
- The notebook `notebooks/experiment.ipynb` hosts the common experiment workflow
- The notebook `notebooks/pipeline.ipynb` packages the training code and compile a scheduled pipeline

# Contact
Please reach out to dvquy.13@gmail.com if you want to discuss anything about this repo.
