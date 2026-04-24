from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager
from backend.lib.job_manager.types import RemoteJobType

from .types import RemoteJobInputPayload, RemoteJobOutputPayload

TInputPayload = TypeVar(
    "TInputPayload", bound=RemoteJobInputPayload, contravariant=True
)  # Bound to Pydantic models
TOutputPayload = TypeVar("TOutputPayload", bound=RemoteJobOutputPayload, covariant=True)


class RemoteJobProcessor(Generic[TInputPayload, TOutputPayload], ABC):
    def __init__(
        self,
        job_uuid: UUID,
        job_type: RemoteJobType,
        asset_manager: AssetManager,
        db_session_factory: AsyncSessionFactory,
    ) -> None:
        self.job_uuid = job_uuid
        self.job_type = job_type
        self.asset_manager = asset_manager
        self.db_session_factory = db_session_factory

    @abstractmethod
    async def process(self, input_payload: TInputPayload) -> TOutputPayload: ...
