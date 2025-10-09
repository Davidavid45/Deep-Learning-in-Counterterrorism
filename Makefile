.PHONY: help install clean preprocess train baseline ablations paper-assets test

help:
	@echo "Available commands:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make clean          - Remove cache and temporary files"
	@echo "  make preprocess     - Run data preprocessing pipeline"
	@echo "  make baseline       - Run all baseline models"
	@echo "  make train          - Train best model (Bidirectional LSTM)"
	@echo "  make ablations      - Run ablation studies"
	@echo "  make paper-assets   - Generate figures and tables for paper"
	@echo "  make test           - Run basic tests"

install:
	pip install -r requirements.txt

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f nohup.out feature_ablation.log
	@echo "Clean complete!"

preprocess:
	@echo "Running preprocessing pipeline..."
	cd src/preprocess && python run_preprocess.py

baseline:
	@echo "Running all baseline models..."
	cd src/baselines && python run_all_baselines.py

train:
	@echo "Training Bidirectional LSTM (best model)..."
	python src/models/bidirectional_lstm_weekly.py

train-lstm-att:
	@echo "Training LSTM with Attention..."
	python src/models/lstm_attention_weekly.py

ablations:
	@echo "Running ablation studies..."
	python src/ablations/run_ablations.py

ablations-feature:
	@echo "Running feature ablation only..."
	python src/ablations/run_ablations.py --feature-only

ablations-data:
	@echo "Running data ablation only..."
	python src/ablations/run_ablations.py --data-only

paper-assets:
	@echo "Generating figures for paper..."
	python scripts/create_paper_figures.py
	@echo "Generating LaTeX tables for paper..."
	python scripts/create_paper_tables.py
	@echo "Paper assets generated in reports/figures/paper/ and reports/tables/latex/"

test:
	@echo "Running basic tests..."
	@python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
	@python -c "import pandas as pd; print('Pandas:', pd.__version__)"
	@python -c "import numpy as np; print('NumPy:', np.__version__)"
	@echo "All dependencies OK!"
