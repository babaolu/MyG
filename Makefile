.PHONY: backend shell dev test lint

backend:
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

shell:
	uv run python shell.py

dev:
	@echo "Starting VulkanMind backend + shell..."
	@uv run uvicorn main:app --reload &
	@sleep 2
	@uv run python shell.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
