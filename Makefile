.PHONY: install train test run docker
install:
	pip install -e ".[dev]"
train:
	python -m creditoptima.training
test:
	pytest -q
run:
	uvicorn creditoptima.api:app --reload
docker:
	docker compose up --build

