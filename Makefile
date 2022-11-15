build-notebook-image:
	cp poetry.lock cloudbuild/notebook/
	cp pyproject.toml cloudbuild/notebook/
	cd cloudbuild/notebook && \
	gcloud builds submit \
		--config cloudbuild_notebook.yaml \
		--timeout 3600s \
		.

build-main-image:
	cp poetry.lock cloudbuild/main/
	cp pyproject.toml cloudbuild/main/
	cp -r seq_rec/ cloudbuild/main/seq_rec/
	cd cloudbuild/main && \
	gcloud builds submit \
		--region=asia-southeast1 \
		--config cloudbuild_main.yaml \
		--timeout 3600s \
		.
