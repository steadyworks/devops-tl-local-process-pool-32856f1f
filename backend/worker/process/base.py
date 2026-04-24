# pyright: reportUnnecessaryComparison=false
import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import (
    AsyncGenerator,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.data_models import JobStatus
from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager
from backend.lib.asset_manager.factory import AssetManagerFactory
from backend.lib.job_manager.base import AbstractJobManager
from backend.lib.redis.factory import RedisClientFactory
from backend.logging_utils import configure_logging_env

NUM_CONCURRENT_WORKER_TASKS = 6
MAX_JOB_TIMEOUT_SECS = 300  # 5 mins
SEND_HEARTBEAT_EVERY_SECS = 1
POLL_SHUTDOWN_EVERY_SECS = 1
HEARTBEAT_PING_MSG = "ping"
SHUTDOWN_SIGNAL_MSG = "shutdown"
JOB_MANAGER_POLL_TIMEOUT_SECS = 5
WORKER_RESTART_BACKOFF_SECS = 1


@runtime_checkable
class CanDumpToJson(Protocol):
    def model_dump_json(self) -> str: ...


TJobType = TypeVar("TJobType")
TJobInput = TypeVar("TJobInput", bound=CanDumpToJson)
TJobOutput = TypeVar("TJobOutput", bound=CanDumpToJson)


@asynccontextmanager
async def maybe_db_session(
    factory: Optional[AsyncSessionFactory],
) -> AsyncGenerator[Optional[AsyncSession], None]:
    if factory is None:
        yield None
    else:
        async with factory.new_session() as session:
            yield session


class BaseWorkerProcess(Process, ABC):
    def __init__(self, heartbeat_connection: Connection, name: str = "worker") -> None:
        super().__init__()
        self.name = name
        self.heartbeat_connection = heartbeat_connection

    @abstractmethod
    def run(self) -> None: ...


class AbstractWorkerProcess(
    BaseWorkerProcess, Generic[TJobType, TJobInput, TJobOutput]
):
    def __init__(self, heartbeat_connection: Connection, name: str = "worker") -> None:
        # Runs in the parent process
        super().__init__(heartbeat_connection, name)

    @abstractmethod
    def _create_redis_client_factory(self) -> RedisClientFactory: ...

    @abstractmethod
    def _get_job_manager_cls(
        self,
    ) -> type[AbstractJobManager[TJobType, TJobInput, TJobOutput]]: ...

    @abstractmethod
    def _get_job_queue_name(self) -> str: ...

    @abstractmethod
    def _create_db_session_factory(self) -> Optional[AsyncSessionFactory]: ...

    @abstractmethod
    async def _process_job(
        self,
        worker_thread_id: int,
        job_uuid: UUID,
        job_type: TJobType,
        job_input_payload: TJobInput,
        asset_manager: AssetManager,
        db_session_factory: Optional[AsyncSessionFactory],
    ) -> TJobOutput: ...

    def _start_heartbeat_ping_thread(
        self,
        shutdown_event: asyncio.Event,
    ) -> None:
        def send_heartbeat(_shutdown_event: asyncio.Event) -> None:
            while not _shutdown_event.is_set():
                try:
                    self.heartbeat_connection.send(HEARTBEAT_PING_MSG)
                    time.sleep(SEND_HEARTBEAT_EVERY_SECS)
                except Exception:
                    logging.warning(
                        f"[{self.name}] Heartbeat pipe closed (send)"
                    )  # parent closed pipe
                    _shutdown_event.set()
                    break

        threading.Thread(
            target=send_heartbeat, args=(shutdown_event,), daemon=True
        ).start()

    def _start_heartbeat_shutdown_monitor_thread(
        self,
        shutdown_event: asyncio.Event,
    ) -> None:
        def monitor_shutdown(_shutdown_event: asyncio.Event) -> None:
            while not _shutdown_event.is_set():
                try:
                    if self.heartbeat_connection.poll(timeout=POLL_SHUTDOWN_EVERY_SECS):
                        msg = self.heartbeat_connection.recv()
                        if msg == SHUTDOWN_SIGNAL_MSG:
                            logging.info(f"[{self.name}] Received shutdown signal")
                            _shutdown_event.set()
                            break
                except (EOFError, OSError):
                    logging.warning(f"[{self.name}] Heartbeat pipe closed")
                    _shutdown_event.set()
                    break
                time.sleep(0.1)

        threading.Thread(
            target=monitor_shutdown, args=(shutdown_event,), daemon=True
        ).start()

    def run(self) -> None:
        # Run in child process
        try:
            configure_logging_env()
            logging.info(f"[{self.name}] Worker started with PID {self.pid}")

            # Initialize shutdown event
            shutdown_event = asyncio.Event()

            # Initialize heartbeat ping and shutdown monitor threads
            self._start_heartbeat_ping_thread(shutdown_event)
            self._start_heartbeat_shutdown_monitor_thread(shutdown_event)

            # Initialize resources or resource factories shared by all tasks
            asset_manager = AssetManagerFactory().create()
            redis_client_factory = self._create_redis_client_factory()
            db_session_factory = self._create_db_session_factory()

            async def _wrapped_main(
                _asset_manager: AssetManager,
                _redis_client_factory: RedisClientFactory,
                _db_session_factory: Optional[AsyncSessionFactory],
                _shutdown_event: asyncio.Event,
            ) -> None:
                try:
                    await self._supervised_main_loop_forever(
                        _asset_manager,
                        _redis_client_factory,
                        _db_session_factory,
                        _shutdown_event,
                    )
                finally:
                    logging.info("Closing pool")
                    await _redis_client_factory.close_pool()

            asyncio.run(
                _wrapped_main(
                    asset_manager,
                    redis_client_factory,
                    db_session_factory,
                    shutdown_event,
                )
            )
        except Exception as e:
            logging.exception(f"[{self.name}] Worker crashed: {e}")

    async def _supervised_main_loop_forever(
        self,
        asset_manager: AssetManager,
        redis_client_factory: RedisClientFactory,
        db_session_factory: Optional[AsyncSessionFactory],
        shutdown_event: asyncio.Event,
    ) -> None:
        logging.info(f"[{self.name}] Started worker process (PID={self.pid})")

        try:
            # Launch all workers + monitor
            await self._run_worker_supervisor_loop(
                asset_manager, redis_client_factory, db_session_factory, shutdown_event
            )
        except asyncio.CancelledError as e:
            logging.info(f"[{self.name}] Supervisor received cancel signal: {e}")
        except Exception as e:
            logging.info(f"[{self.name}] Supervisor unexpected exception: {e}")

        logging.info(f"[{self.name}] All tasks shut down cleanly")

    async def _run_worker_supervisor_loop(
        self,
        asset_manager: AssetManager,
        redis_client_factory: RedisClientFactory,
        db_session_factory: Optional[AsyncSessionFactory],
        shutdown_event: asyncio.Event,
    ) -> None:
        running_tasks: dict[int, asyncio.Task[None]] = {}

        # Start all workers
        for i in range(NUM_CONCURRENT_WORKER_TASKS):
            running_tasks[i] = asyncio.create_task(
                self._spawn_worker_forever(
                    i,
                    asset_manager,
                    redis_client_factory,
                    db_session_factory,
                    shutdown_event,
                )
            )

        # Monitor loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
            for i, task in list(running_tasks.items()):
                if task.done():
                    exc = task.exception()
                    if exc:
                        logging.error(
                            f"[{self.name}] Worker-{i} exited with error: {exc}"
                        )
                    else:
                        logging.warning(
                            f"[{self.name}] Worker-{i} exited cleanly (unexpected)"
                        )

                    # Restart
                    logging.info(f"[{self.name}] Restarting Worker-{i}")
                    if not shutdown_event.is_set():
                        running_tasks[i] = asyncio.create_task(
                            self._spawn_worker_forever(
                                i,
                                asset_manager,
                                redis_client_factory,
                                db_session_factory,
                                shutdown_event,
                            )
                        )

        # Shutdown triggered: cancel all
        logging.info(f"[{self.name}] Cancelling all workers...")
        for task in running_tasks.values():
            task.cancel()

        await asyncio.gather(*running_tasks.values(), return_exceptions=True)
        logging.info(f"[{self.name}] All workers shut down cleanly")

    async def _spawn_worker_forever(
        self,
        i: int,
        asset_manager: AssetManager,
        redis_client_factory: RedisClientFactory,
        db_session_factory: Optional[AsyncSessionFactory],
        shutdown_event: asyncio.Event,
    ) -> None:
        job_manager_cls = self._get_job_manager_cls()
        async with job_manager_cls(
            redis_client_factory, self._get_job_queue_name()
        ) as job_manager:
            while not shutdown_event.is_set():
                try:
                    logging.info(f"[{self.name}-thread_{i}] Spawning worker-{i}")
                    await self._job_worker_main_loop(
                        i,
                        job_manager,
                        asset_manager,
                        db_session_factory,
                        shutdown_event,
                    )
                except Exception as e:
                    logging.exception(
                        f"[{self.name}-thread_{i}] Worker-{i} crashed: {e}. Restarting after delay."
                    )
                    await asyncio.sleep(WORKER_RESTART_BACKOFF_SECS)  # optional backoff

    async def _job_worker_main_loop(
        self,
        worker_thread_id: int,
        job_manager: AbstractJobManager[TJobType, TJobInput, TJobOutput],
        asset_manager: AssetManager,
        db_session_factory: Optional[AsyncSessionFactory],
        shutdown_event: asyncio.Event,
    ) -> None:
        while not shutdown_event.is_set():
            try:
                logging.debug(
                    f"[{self.name}-thread_{worker_thread_id}] Polling from redis"
                )
                job_uuid = await job_manager.poll(timeout=JOB_MANAGER_POLL_TIMEOUT_SECS)
                logging.debug(
                    f"[{self.name}-thread_{worker_thread_id}] Received job from redis: {job_uuid}"
                )

                if shutdown_event.is_set():
                    break
                if job_uuid is None:
                    continue

                await self._process_job_polled_from_redis(
                    worker_thread_id,
                    job_uuid,
                    job_manager,
                    asset_manager,
                    db_session_factory,
                )
            except asyncio.CancelledError:
                logging.info(f"[{self.name}-thread_{worker_thread_id}] Cancelled")
                raise
            except Exception:
                logging.exception(
                    f"[{self.name}-thread_{worker_thread_id}] Unexpected error"
                )

    async def _process_job_polled_from_redis(
        self,
        worker_thread_id: int,
        job_uuid: UUID,
        job_manager: AbstractJobManager[TJobType, TJobInput, TJobOutput],
        asset_manager: AssetManager,
        db_session_factory: Optional[AsyncSessionFactory],
    ) -> None:
        job_type, job_input_payload = None, None
        try:
            async with maybe_db_session(db_session_factory) as db_session:
                job_type, job_input_payload = await job_manager.claim(
                    job_uuid, db_session=db_session
                )
                if job_type is None or job_input_payload is None:
                    logging.warning(
                        f"[{self.name}-thread_{worker_thread_id}] Received incomplete job {job_uuid}"
                    )
                    await self._mark_job_as_error(
                        worker_thread_id,
                        job_manager,
                        db_session_factory,
                        job_uuid,
                        "Claimed job missing type or input payload",
                    )
                    return
        except asyncio.CancelledError:
            logging.info(
                f"[{self.name}-thread_{worker_thread_id}] Cancelled while claiming job {job_uuid}"
            )
            raise
        except Exception:
            logging.exception(
                f"[{self.name}-thread_{worker_thread_id}] Job claim DB write failed for job: {job_uuid}"
            )
            await self._mark_job_as_error(
                worker_thread_id,
                job_manager,
                db_session_factory,
                job_uuid,
                "Failed to mark job as dequeued",
            )
            return  # Not successfully claimed

        try:
            await asyncio.wait_for(
                self._handle_task(
                    worker_thread_id,
                    job_uuid,
                    job_type,
                    job_input_payload,
                    job_manager,
                    asset_manager,
                    db_session_factory,
                ),
                timeout=MAX_JOB_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            logging.info(
                f"[{self.name}-thread_{worker_thread_id}] Cancelled while running job {job_uuid}"
            )
            raise
        except asyncio.TimeoutError:
            logging.warning(
                f"[{self.name}-thread_{worker_thread_id}] Job timed out after {MAX_JOB_TIMEOUT_SECS}s, "
                f"job_id: {job_uuid} "
                f"payload: {job_input_payload.model_dump_json() if job_input_payload else '<missing payload>'}"
            )
            await self._mark_job_as_error(
                worker_thread_id,
                job_manager,
                db_session_factory,
                job_uuid,
                f"Timeout after {MAX_JOB_TIMEOUT_SECS}s",
            )
        except Exception as e:
            logging.warning(
                f"[{self.name}-thread_{worker_thread_id}] Job failed: job_id: {job_uuid} payload: "
                f"payload: {job_input_payload.model_dump_json() if job_input_payload else '<missing payload>'}"
            )
            await self._mark_job_as_error(
                worker_thread_id,
                job_manager,
                db_session_factory,
                job_uuid,
                f"Job execution failed due to {str(e)}",
            )

    async def _handle_task(
        self,
        worker_thread_id: int,
        job_uuid: UUID,
        job_type: TJobType,
        job_input_payload: TJobInput,
        job_manager: AbstractJobManager[TJobType, TJobInput, TJobOutput],
        asset_manager: AssetManager,
        db_session_factory: Optional[AsyncSessionFactory],
    ) -> None:
        try:
            async with maybe_db_session(db_session_factory) as db_session:
                await job_manager.update_status(
                    job_uuid, JobStatus.PROCESSING, db_session=db_session
                )
            result = await self._process_job(
                worker_thread_id,
                job_uuid,
                job_type,
                job_input_payload,
                asset_manager,
                db_session_factory,
            )
            async with maybe_db_session(db_session_factory) as db_session:
                await job_manager.update_status(
                    job_uuid,
                    JobStatus.DONE,
                    result_payload=result,
                    db_session=db_session,
                )
        except asyncio.CancelledError:
            logging.info(
                f"[{self.name}-thread_{worker_thread_id}] Cancelled while processing job {job_uuid}"
            )
            raise
        except Exception as e:
            logging.warning(
                f"[{self.name}-thread_{worker_thread_id}] Failed job {job_uuid}: {e}"
            )
            raise e

    async def _mark_job_as_error(
        self,
        worker_thread_id: int,
        job_manager: AbstractJobManager[TJobType, TJobInput, TJobOutput],
        db_session_factory: Optional[AsyncSessionFactory],
        job_uuid: UUID,
        reason: str,
    ) -> None:
        try:
            async with maybe_db_session(db_session_factory) as db_session:
                await job_manager.update_status(
                    job_uuid,
                    JobStatus.ERROR,
                    error_message=reason,
                    db_session=db_session,
                )
        except Exception as inner:
            logging.warning(
                f"[{self.name}-thread_{worker_thread_id}] Failed to mark job {job_uuid} as error: {inner}"
            )
