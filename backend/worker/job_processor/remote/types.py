from uuid import UUID

from pydantic import BaseModel


class RemoteJobInputPayload(BaseModel):
    user_id: UUID
    originating_photobook_id: UUID


class RemoteJobOutputPayload(BaseModel):
    job_id: UUID


class PhotobookGenerationInputPayload(RemoteJobInputPayload):
    asset_ids: list[UUID]


class PhotobookGenerationOutputPayload(RemoteJobOutputPayload):
    pass
