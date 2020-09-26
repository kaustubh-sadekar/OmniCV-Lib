PYTHON = python3

.PHONY = help setup test run clean
.DEFAULT_GOAL = help

help:
	@echo "---------------HELP-----------------"
	@echo "To build only python extension type make build-py"
	@echo "To build only C++ extension type make build-cpp"
	@echo "To build both python and C++ extensions type make build-all"
	@echo "To test only python extension of the project type make test-py"
	@echo "To run visual tests for only python extension type make test-py-gui"
	@echo "To test only C++ extension of the project type make test-cpp"
	@echo "To run visual tests for only C++ extension type make test-cpp-gui"
	@echo "To test Python and C++ extension of the project type make test-all"
	@echo "To run visual tests for Python and C++ extension type make test-all-gui"
	@echo "------------------------------------"

build-py:
	@echo "=============================================== \n"
	@echo "Installing omnicv python library from source"
	@echo "=============================================== \n"
	cd ../ && sudo ${PYTHON} setup.py install
	@echo "Running tests for Python extension \n"
	@echo "=============================================== \n"
	$(MAKE) test-py

test-py:
	@echo "=============================================== \n"
	@echo "Running tests for Python extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ${PYTHON} -W ignore -m pytest --ignore=./test_visual.py

test-py-gui:
	@echo "=============================================== \n"
	@echo "Running visual tests for Python extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ${PYTHON} -W ignore -m pytest -s --ignore=./test_.py && ./test
	
build-cpp:
	@echo "=============================================== \n"
	@echo "Building omnicv c++ files"
	@echo "=============================================== \n"
	cd ../build/ && cmake -Wno-dev CMakeLists.txt && $(MAKE)
	$(MAKE) test-cpp

test-cpp:
	@echo "=============================================== \n"
	@echo "Running tests for C++ extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ./test

test-cpp-gui:
	@echo "=============================================== \n"
	@echo "Running visual tests for C++ extension \n"
	@echo "=============================================== \n"
	cd ../tests/ && ./test_visual

build-all:
	$(MAKE) build-py
	$(MAKE) build-cpp

test-all:
	$(MAKE) test-py
	$(MAKE) test-cpp

test-all-gui:
	$(MAKE) test-py-gui
	$(MAKE) test-cpp-gui
