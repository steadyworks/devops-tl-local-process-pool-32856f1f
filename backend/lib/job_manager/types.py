from enum import Enum


class RemoteJobQueue(Enum):
    MAIN_TASK_QUEUE = "main_task_queue"


class RemoteJobType(str, Enum):
    PHOTOBOOK_GENERATION = "photobook_generation"


class LocalJobQueue(Enum):
    MAIN_TASK_QUEUE_LOCAL = "main_task_queue_local"


class LocalJobType(str, Enum):
    PHOTOBOOK_GENERATION = "photobook_generation"
