import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-integration", action="store_true", default=False, help="Skip integration tests."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-integration"):
        # --skip-integration given in cli: skip these tests
        skip_slow = pytest.mark.skip(
            reason="skipped due to --skip-integration option"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_slow)
