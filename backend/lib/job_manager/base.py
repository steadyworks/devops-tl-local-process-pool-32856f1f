import asyncio
import logging
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Generic, Optional, Self, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.data_models import JobStatus
from backend.lib.redis.factory import RedisClientFactory, SafeRedisClient

# Define base job payload and output payload types
TJobType = TypeVar("TJobType")
TJobInput = TypeVar("TJobInput")
TJobOutput = TypeVar("TJobOutput")

DEFAULT_DEQUEUE_POLL_TIMEOUT_SECS = 5


class AbstractJobManager(ABC, Generic[TJobType, TJobInput, TJobOutput]):
    """
    Usage:

    # redis_client_factory can be shared across threads
    # job_manager is not thread or asyncio task safe (one per thread/task)

    with JobManager(redis_client_factory, queue_name) as job_manager:
        ...
    """

    def __init__(
        self, redis_client_factory: RedisClientFactory, queue_name: str
    ) -> None:
        self.redis_client_factory = redis_client_factory
        self.queue_name = queue_name

    async def __aenter__(self) -> Self:
        self.redis_client: SafeRedisClient = (
            self.redis_client_factory.new_redis_client()
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.redis_client:
            await self.redis_client.close()

    async def poll(
        self,
        timeout: int,
    ) -> Optional[UUID]:
        try:
            result = await asyncio.wait_for(
                self.redis_client.safe_blpop(self.queue_name, timeout=timeout),
                timeout=timeout + 5,  # pad to account for reconnect etc.
            )
        except asyncio.TimeoutError:
            logging.warning(
                f"[job manager] Unexpected poll timeout after {timeout} secs..."
            )
            # Redis may be stuck, kill the task and let supervisor handle retry
            return None
        except Exception as e:
            logging.exception(f"[job manager] Unexpected exception occurred: {e}")
            raise e

        if not result:
            return None  # timeout occurred

        _queue_name, job_id_str = result
        try:
            return UUID(job_id_str)
        except ValueError:
            return None

    @abstractmethod
    async def enqueue(
        self,
        job_type: TJobType,
        job_payload: TJobInput,
        db_session: Optional[AsyncSession] = None,
    ) -> UUID: ...

    @abstractmethod
    async def claim(
        self,
        job_id: UUID,
        db_session: Optional[AsyncSession] = None,
    ) -> tuple[TJobType, TJobInput]: ...

    @abstractmethod
    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        error_message: Optional[str] = None,
        result_payload: Optional[TJobOutput] = None,
        db_session: Optional[AsyncSession] = None,
    ) -> None: ...
