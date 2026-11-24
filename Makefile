.PHONY: install lint type test check eval clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

type:
	uv run mypy

test:
	uv run pytest

# What CI runs. Never makes a billed model call.
check: lint type test

# Phase 1 deliverable: the metric table for the evaluation arms.
eval:
	uv run scriptorium eval --config config/eval.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
