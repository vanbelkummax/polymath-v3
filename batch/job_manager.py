"""
GCP Batch API job manager for Polymath v3.

Handles job submission, monitoring, and cleanup for batch processing.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from lib.config import config

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a batch job."""

    job_id: str
    status: JobStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_uri: Optional[str] = None
    error_message: Optional[str] = None
    items_processed: int = 0
    items_failed: int = 0


class BatchJobManager:
    """
    Manage GCP Batch API jobs.

    Usage:
        manager = BatchJobManager()
        job_id = manager.submit_job(
            job_type="concept_extraction",
            input_uri="gs://bucket/input.jsonl",
            output_uri="gs://bucket/output.jsonl",
        )
        result = manager.wait_for_completion(job_id)
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: str = "us-central1",
    ):
        """
        Initialize job manager.

        Args:
            project_id: GCP project ID (from config if not provided)
            region: GCP region for batch jobs
        """
        self.project_id = project_id or config.GCP_PROJECT_ID
        self.region = region
        self._client = None

    @property
    def client(self):
        """Lazy load GCP Batch client."""
        if self._client is None:
            try:
                from google.cloud import batch_v1

                self._client = batch_v1.BatchServiceClient()
            except ImportError:
                raise ImportError("google-cloud-batch package not installed")

        return self._client

    def submit_job(
        self,
        job_type: str,
        input_uri: str,
        output_uri: str,
        machine_type: str = "e2-standard-4",
        task_count: int = 1,
        parallelism: int = 1,
        timeout: str = "3600s",
        labels: Optional[dict] = None,
    ) -> str:
        """
        Submit a batch job.

        Args:
            job_type: Type of job (concept_extraction, embedding, etc.)
            input_uri: GCS URI for input data
            output_uri: GCS URI for output data
            machine_type: GCE machine type
            task_count: Number of tasks
            parallelism: Tasks to run in parallel
            timeout: Job timeout
            labels: Optional labels for the job

        Returns:
            Job ID
        """
        from google.cloud import batch_v1
        from google.protobuf import duration_pb2

        # Generate job ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id = f"polymath-{job_type}-{timestamp}"

        # Define the task
        task = batch_v1.TaskSpec()

        # Container for the task
        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container(
            image_uri=f"gcr.io/{self.project_id}/polymath-batch:latest",
            commands=[
                "python",
                "-m",
                f"batch.{job_type}",
                "--input-uri",
                input_uri,
                "--output-uri",
                output_uri,
            ],
        )
        task.runnables = [runnable]

        # Resources
        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 4000  # 4 vCPUs
        resources.memory_mib = 16384  # 16 GB
        task.compute_resource = resources

        # Max run duration
        task.max_run_duration = duration_pb2.Duration(seconds=int(timeout.rstrip("s")))

        # Task group
        task_group = batch_v1.TaskGroup()
        task_group.task_spec = task
        task_group.task_count = task_count
        task_group.parallelism = parallelism

        # Allocation policy
        allocation_policy = batch_v1.AllocationPolicy()
        allocation_policy.instances = [
            batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                policy=batch_v1.AllocationPolicy.InstancePolicy(
                    machine_type=machine_type,
                )
            )
        ]

        # Create job
        job = batch_v1.Job()
        job.task_groups = [task_group]
        job.allocation_policy = allocation_policy
        job.logs_policy = batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        )

        if labels:
            job.labels = labels

        # Submit
        parent = f"projects/{self.project_id}/locations/{self.region}"
        request = batch_v1.CreateJobRequest(
            parent=parent,
            job_id=job_id,
            job=job,
        )

        operation = self.client.create_job(request=request)
        logger.info(f"Submitted job: {job_id}")

        # Track in database
        self._record_job(job_id, job_type, input_uri, output_uri)

        return job_id

    def get_job_status(self, job_id: str) -> JobResult:
        """
        Get status of a job.

        Args:
            job_id: Job ID

        Returns:
            JobResult with current status
        """
        from google.cloud import batch_v1

        name = f"projects/{self.project_id}/locations/{self.region}/jobs/{job_id}"

        try:
            job = self.client.get_job(name=name)

            status_map = {
                batch_v1.JobStatus.State.STATE_UNSPECIFIED: JobStatus.PENDING,
                batch_v1.JobStatus.State.QUEUED: JobStatus.PENDING,
                batch_v1.JobStatus.State.SCHEDULED: JobStatus.PENDING,
                batch_v1.JobStatus.State.RUNNING: JobStatus.RUNNING,
                batch_v1.JobStatus.State.SUCCEEDED: JobStatus.SUCCEEDED,
                batch_v1.JobStatus.State.FAILED: JobStatus.FAILED,
                batch_v1.JobStatus.State.DELETION_IN_PROGRESS: JobStatus.CANCELLED,
            }

            status = status_map.get(job.status.state, JobStatus.PENDING)

            return JobResult(
                job_id=job_id,
                status=status,
                started_at=job.create_time.timestamp() if job.create_time else None,
                items_processed=sum(
                    tc.succeeded_count for tc in job.status.task_groups
                ),
                items_failed=sum(tc.failed_count for tc in job.status.task_groups),
            )

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> JobResult:
        """
        Wait for job completion.

        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time

        Returns:
            Final JobResult
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.get_job_status(job_id)

            if result.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                self._update_job_record(job_id, result)
                return result

            logger.info(f"Job {job_id}: {result.status.value}")
            time.sleep(poll_interval)

        return JobResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            error_message="Timeout waiting for job completion",
        )

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled successfully
        """
        from google.cloud import batch_v1

        name = f"projects/{self.project_id}/locations/{self.region}/jobs/{job_id}"

        try:
            request = batch_v1.DeleteJobRequest(name=name)
            self.client.delete_job(request=request)
            logger.info(f"Cancelled job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def _record_job(
        self, job_id: str, job_type: str, input_uri: str, output_uri: str
    ):
        """Record job in database."""
        from lib.db.postgres import get_pg_pool

        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO gcp_batch_jobs (
                        job_id, job_type, input_uri, output_uri, status
                    ) VALUES (%s, %s, %s, %s, 'pending')
                    """,
                    (job_id, job_type, input_uri, output_uri),
                )
                conn.commit()

    def _update_job_record(self, job_id: str, result: JobResult):
        """Update job record in database."""
        from lib.db.postgres import get_pg_pool

        pool = get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE gcp_batch_jobs
                    SET status = %s,
                        items_processed = %s,
                        items_failed = %s,
                        completed_at = NOW(),
                        error_message = %s
                    WHERE job_id = %s
                    """,
                    (
                        result.status.value,
                        result.items_processed,
                        result.items_failed,
                        result.error_message,
                        job_id,
                    ),
                )
                conn.commit()
