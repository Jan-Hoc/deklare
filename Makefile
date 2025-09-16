# ToDo: add other endpoints mentioned in readme and adapt readme (maybe add precommit endpoint)
# ToDo: add to readme that you need to have build-essential installed to run make
.PHONY: help setup format lint

help:
	@echo "Available targets:"
	@printf "  %-12s %s\n" "help:" "Print available make targets"
	@printf "  %-12s %s\n" "setup:" "Install all dependencies using uv in virtual environment"
	@printf "  %-12s %s\n" "format:" "Format code using ruff"
	@printf "  %-12s %s\n" "lint:" "Run mypy linting"

setup:
	uv run pre-commit install

format:
	uv run ruff format src
	uv run ruff check --fix

lint:
	uv run mypy --install-types --non-interactive src/deklare
