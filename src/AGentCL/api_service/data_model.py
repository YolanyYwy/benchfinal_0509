from pydantic import BaseModel

from AGentCL.data_model.tasks import Task


class GetTasksRequest(BaseModel):
    """
    Request for getting tasks
    """

    domain: str


class GetTasksResponse(BaseModel):
    """
    Response for getting tasks
    """

    tasks: list[Task]
