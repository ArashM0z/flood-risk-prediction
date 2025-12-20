.PHONY: install dev lint test demo train docker clean
install: ; pip install -e .
dev: ; pip install -e ".[dev]"
lint: ; ruff check src tests
test: ; WANDB_MODE=disabled pytest --cov=flood --cov-report=term-missing
demo: ; WANDB_MODE=disabled flood-train --config configs/default.yaml --demo
train: ; flood-train --config configs/default.yaml
docker: ; docker build -t flood-risk:latest .
clean: ; rm -rf build dist *.egg-info .pytest_cache .ruff_cache mlruns wandb
