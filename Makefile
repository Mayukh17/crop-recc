.PHONY: setup run run-recommender clean

# Python version to use
PYTHON = python3
VENV = venv
BIN = $(VENV)/bin

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

run: setup
	$(BIN)/python crop_recommendation.py

run-recommender: setup
	$(BIN)/python crop_recommender.py

launch:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".env" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type d -name "venv" -exec rm -rf {} +
	find . -type d -name ".idea" -exec rm -rf {} +
	find . -type d -name ".vscode" -exec rm -rf {} +
	find . -type f -name "*.log" -delete