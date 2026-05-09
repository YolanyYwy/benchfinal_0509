"""Utility functions and constants for the ota domain."""

from AGentCL.utils.utils import DATA_DIR

OTA_DATA_DIR = DATA_DIR / "tau2" / "domains" / "ota"
OTA_DB_PATH = OTA_DATA_DIR / "db.json"
OTA_POLICY_PATH = OTA_DATA_DIR / "policy.md"
OTA_TASK_SET_PATH = OTA_DATA_DIR / "tasks.json"
