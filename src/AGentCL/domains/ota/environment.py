"""Environment setup for the ota domain."""

from pathlib import Path
from typing import Optional

from AGentCL.data_model.tasks import Task
from AGentCL.domains.ota.data_model import OTADB
from AGentCL.domains.ota.tools import OTATools
from AGentCL.domains.ota.utils import (
    OTA_DB_PATH,
    OTA_POLICY_PATH,
    OTA_TASK_SET_PATH,
)
from AGentCL.environment.environment import Environment
from AGentCL.utils import load_file


def get_environment(db: Optional[OTADB] = None, solo_mode: bool = False) -> Environment:
    """
    Get the ota domain environment.

    Args:
        db: Optional database to use. If None, loads from default path.
        solo_mode: Whether to run in solo mode (not supported for ota)

    Returns:
        Environment instance for the ota domain
    """
    if solo_mode:
        raise ValueError("OTA domain does not support solo mode")

    if db is None:
        db = OTADB.load(OTA_DB_PATH)

    tools = OTATools(db)

    with open(OTA_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()

    return Environment(
        domain_name="ota",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Get tasks for the ota domain.

    Args:
        task_split_name: Name of the task split to load (base, hard, easy, etc.)

    Returns:
        List of Task objects
    """
    tasks = load_file(OTA_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]

    if task_split_name is None:
        return tasks

    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(f"Invalid task split name: {task_split_name}")

    task_ids = set(task_splits[task_split_name])
    return [task for task in tasks if task.id in task_ids]


def get_tasks_split() -> dict[str, list[str]]:
    """
    Get task splits for the ota domain.

    Returns:
        Dictionary mapping split names to lists of task IDs
    """
    split_file = Path(OTA_TASK_SET_PATH).parent / f"split_{Path(OTA_TASK_SET_PATH).stem}.json"
    if not split_file.exists():
        # Return default split with all tasks in base
        tasks = load_file(OTA_TASK_SET_PATH)
        all_task_ids = [task["id"] for task in tasks]
        return {"base": all_task_ids}
    return load_file(split_file)
