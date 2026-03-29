PYTHON ?= /Users/bnl28/miniforge3/envs/work26/bin/python
PYTEST ?= $(PYTHON) -m pytest
TEST_ENV ?= tests/testenv.sh

.PHONY: help test test-unit test-integration acid-test

help:
	@printf '%s\n' \
		'make test              Run the default non-integration pytest suite' \
		'make test-unit         Alias for make test' \
		'make test-integration  Run pytest integration tests using $(TEST_ENV)' \
		'make acid-test         Run the standalone SSH acid test using $(TEST_ENV)'

test: test-unit

test-unit:
	$(PYTEST) -m "not integration"

test-integration:
	. "$(TEST_ENV)" && $(PYTEST) -m integration

acid-test:
	. "$(TEST_ENV)" && $(PYTHON) tests/acid_test.py