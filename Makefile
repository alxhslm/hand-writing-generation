ENV_FILE := .env

init:
	poetry install

test:
	poetry run pytest
