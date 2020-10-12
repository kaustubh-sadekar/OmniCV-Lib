PYTHON = python3
.DEFAULT_GOAL = help

.PHONY: help
help:
	@echo "---------------HELP-----------------"
	@echo "To test for linting errors type make lint"
	@echo "To build only python extension type make build-py"
	@echo "To build only C++ extension type make build-cpp"
	@echo "To build both python and C++ extensions type make build-all"
	@echo "To test only python extension of the project type make test-py"
	@echo "To test only C++ extension of the project type make test-cpp"
	@echo "To test Python and C++ extension of the project type make test-all"
	@echo "------------------------------------"

.PHONY: lint
lint:
	@echo "================ \n"
	@echo "Lint with flake8 \n"
	@echo "================ \n"
	flake8 --per-file-ignores="__init__.py:F403,F401" && cd ../ && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics && flake8 Examples/ && flake8 applications

.PHONY: build-py
build-py:
	@echo "=============================================== \n"
	@echo "Installing omnicv python library from source"
	@echo "=============================================== \n"
	cd ../ && sudo ${PYTHON} setup.py install
	@echo "Running tests for Python extension \n"
	@echo "=============================================== \n"
	$(MAKE) test-py

.PHONY: test-py
test-py:
	@echo "=============================================== \n"
	@echo "Running tests for Python extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ${PYTHON} -W ignore -m pytest

.PHONY: build-cpp
build-cpp:
	@echo "=============================================== \n"
	@echo "Building omnicv c++ files"
	@echo "=============================================== \n"
	cd ../build/ && cmake -Wno-dev CMakeLists.txt && $(MAKE)
	$(MAKE) test-cpp

.PHONY: test-cpp
test-cpp:
	@echo "=============================================== \n"
	@echo "Running tests for C++ extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ./test

.PHONY: build-all
build-all:
	$(MAKE) build-py
	$(MAKE) build-cpp

.PHONY: test-all
test-all:
	$(MAKE) test-py
	$(MAKE) test-cpp
