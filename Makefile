.PHONY: help install train notebook clean

help:
	@echo "Targets: install | train | notebook | clean"

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

notebook:
	jupyter notebook notebooks/personal_loan_prediction.ipynb

clean:
	rm -rf reports/figures reports/metrics.json models/*.joblib
