FROM gcr.io/deeplearning-platform-release/tf-gpu:latest

# Install CUDA to use GPU
RUN conda install -c conda-forge -y cudatoolkit=11.2 cudnn=8.1.0
# This line to help enable GPU support
# /opt/conda should be replaced by $CONDA_PREFIX actually
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/lib/

ENV APP_DIR=/home/setup
ENV POETRY_VERSION=1.1.12

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR $APP_DIR
COPY ./poetry.lock poetry.lock
COPY ./pyproject.toml pyproject.toml

# Project initialization
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi --no-root

COPY ./seq_rec/ seq_rec/
