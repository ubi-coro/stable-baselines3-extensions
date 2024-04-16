SHELL=/bin/bash
LINT_PATHS=stable_baselines3_extensions/ tests/ setup.py

pytest:
	python3 -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive" -W ignore::tqdm.TqdmExperimentalWarning

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}
