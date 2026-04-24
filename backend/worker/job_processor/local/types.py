from dataclasses import dataclass
from uuid import UUID


@dataclass
class LocalJobInputPayload:
    pass


class LocalJobOutputPayload:
    job_id: UUID
