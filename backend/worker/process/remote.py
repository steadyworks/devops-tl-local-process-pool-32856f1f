import logging
from typing import Optional
from uuid import UUID

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager
from backend.lib.job_manager.base import AbstractJobManager
from backend.lib.job_manager.remote import RemoteJobManager
from backend.lib.job_manager.types import RemoteJobQueue, RemoteJobType
from backend.lib.redis.factory import RedisClientFactory
from backend.lib.utils.common import none_throws
from backend.worker.job_processor.remote.factory import RemoteJobProcessorFactory
from backend.worker.job_processor.remote.types import (
    RemoteJobInputPayload,
    RemoteJobOutputPayload,
)

from .base import AbstractWorkerProcess


class RemoteJobWorkerProcess(
    AbstractWorkerProcess[RemoteJobType, RemoteJobInputPayload, RemoteJobOutputPayload]
):
    def _create_redis_client_factory(self) -> RedisClientFactory:
        return RedisClientFactory.from_remote_defaults()

    def _get_job_manager_cls(
        self,
    ) -> type[
        AbstractJobManager[RemoteJobType, RemoteJobInputPayload, RemoteJobOutputPayload]
    ]:
        return RemoteJobManager

    def _get_job_queue_name(self) -> str:
        return RemoteJobQueue.MAIN_TASK_QUEUE.value

    def _create_db_session_factory(self) -> Optional[AsyncSessionFactory]:
        return AsyncSessionFactory()

    async def _process_job(
        self,
        worker_thread_id: int,
        job_uuid: UUID,
        job_type: RemoteJobType,
        job_input_payload: RemoteJobInputPayload,
        asset_manager: AssetManager,
        db_session_factory: Optional[AsyncSessionFactory],
    ) -> RemoteJobOutputPayload:
        logging.info(
            f"[{self.name}-thread_{worker_thread_id}] Processing {job_uuid}, job type: {job_type}"
            f", job_input_payload: {job_input_payload.model_dump_json()}"
        )
        job_processor = RemoteJobProcessorFactory.new_processor(
            job_uuid, job_type, asset_manager, none_throws(db_session_factory)
        )
        result = await job_processor.process(job_input_payload)
        logging.info(
            f"[{self.name}-thread_{worker_thread_id}] Processed {job_uuid}, job type: {job_type}"
        )
        return result
