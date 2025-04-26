.PHONY: setup run run-recommender launch clean

# Python version to use
PYTHON = python3
VENV = venv
BIN = $(VENV)/bin

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	touch $(VENV)/bin/activate

run: setup
	$(BIN)/python crop_recommendation.py

run-recommender: setup
	$(BIN)/python crop_recommender.py

install:
	$(BIN)/pip install -r requirements.txt

dump:
	$(BIN)/pip freeze > requirements.txt

launch: setup
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -o -name ".coverage" -o -name "*.log" \) -delete
	find . -type d \( -name "*.egg-info" -o -name "*.egg" -o -name ".pytest_cache" -o -name "htmlcov" -o -name ".tox" -o -name ".env" -o -name ".venv" -o -name "venv" -o -name ".idea" -o -name ".vscode" \) -exec rm -rf {} +
