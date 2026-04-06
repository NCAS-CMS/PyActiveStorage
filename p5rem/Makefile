PYTHON ?= /Users/bnl28/miniforge3/envs/work26/bin/python
PYTHON_CFDM ?= /Users/bnl28/miniforge3/envs/work26t/bin/python
PYTEST ?= $(PYTHON) -m pytest
TEST_ENV ?= tests/testenv.sh

.PHONY: help test test-unit test-integration test-cfdm acid-test

help:
	@printf '%s\n' \
		'make test              Run the default non-integration pytest suite' \
		'make test-unit         Alias for make test' \
		'make test-integration  Run pytest integration tests using $(TEST_ENV)' \
		'make test-cfdm         Run cfdm compatibility tests with $(PYTHON_CFDM)' \
		'make acid-test         Run the standalone SSH acid test using $(TEST_ENV)'

test: test-unit

test-unit:
	$(PYTEST) -m "not integration"

test-integration:
	. "$(TEST_ENV)" && $(PYTEST) -m integration

test-cfdm:
	$(PYTHON_CFDM) -m pytest tests/test_cfdm_local.py -q

acid-test:
	. "$(TEST_ENV)" && $(PYTHON) tests/acid_test.py