"""Environment setup for the instore domain."""

from pathlib import Path
from typing import Optional

from AGentCL.data_model.tasks import Task
from AGentCL.domains.instore.data_model import InstoreDB
from AGentCL.domains.instore.tools import InstoreTools
from AGentCL.domains.instore.utils import (
    INSTORE_DB_PATH,
    INSTORE_POLICY_PATH,
    INSTORE_TASK_SET_PATH,
)
from AGentCL.environment.environment import Environment
from AGentCL.utils import load_file


def get_environment(db: Optional[InstoreDB] = None, solo_mode: bool = False) -> Environment:
    """
    Get the instore domain environment.

    Args:
        db: Optional database to use. If None, loads from default path.
        solo_mode: Whether to run in solo mode (not supported for instore)

    Returns:
        Environment instance for the instore domain
    """
    if solo_mode:
        raise ValueError("Instore domain does not support solo mode")

    if db is None:
        db = InstoreDB.load(INSTORE_DB_PATH)

    tools = InstoreTools(db)

    with open(INSTORE_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()

    return Environment(
        domain_name="instore",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """
    Get tasks for the instore domain.

    Args:
        task_split_name: Name of the task split to load (base, hard, easy, etc.)

    Returns:
        List of Task objects
    """
    tasks = load_file(INSTORE_TASK_SET_PATH)
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
    Get task splits for the instore domain.

    Returns:
        Dictionary mapping split names to lists of task IDs
    """
    split_file = Path(INSTORE_TASK_SET_PATH).parent / f"split_{Path(INSTORE_TASK_SET_PATH).stem}.json"
    if not split_file.exists():
        # Return default split with all tasks in base
        tasks = load_file(INSTORE_TASK_SET_PATH)
        all_task_ids = [task["id"] for task in tasks]
        return {"base": all_task_ids}
    return load_file(split_file)
