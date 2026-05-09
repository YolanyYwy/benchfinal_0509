"""Utility functions and constants for the delivery domain."""

from AGentCL.utils.utils import DATA_DIR

DELIVERY_DATA_DIR = DATA_DIR / "tau2" / "domains" / "delivery"
DELIVERY_DB_PATH = DELIVERY_DATA_DIR / "db.json"
DELIVERY_POLICY_PATH = DELIVERY_DATA_DIR / "policy.md"
DELIVERY_TASK_SET_PATH = DELIVERY_DATA_DIR / "tasks.json"
