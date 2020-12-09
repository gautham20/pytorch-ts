BIN=.venv/bin

remove-env:
	rm -fr .venv/

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .cache/
	rm -fr .venv/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.so' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean: remove-env clean-build clean-pyc clean-test

virtualenv:
	python3 -m venv .venv

test:
	$(BIN)/pytest -v tests

install: clean virtualenv
	$(BIN)/pip install -r requirements.txt -U

reinstall:
	$(BIN)/pip install -r requirements.txt -U

pip_list:
	$(BIN)/pip list > pip_list
