"""Utility functions and constants for the instore domain."""

from AGentCL.utils.utils import DATA_DIR

INSTORE_DATA_DIR = DATA_DIR / "tau2" / "domains" / "instore"
INSTORE_DB_PATH = INSTORE_DATA_DIR / "db.json"
INSTORE_POLICY_PATH = INSTORE_DATA_DIR / "policy.md"
INSTORE_TASK_SET_PATH = INSTORE_DATA_DIR / "tasks.json"
