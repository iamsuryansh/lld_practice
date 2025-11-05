'''
Design a Distributed Job Processing System (Low Level Design)

FUNCTIONAL REQUIREMENTS:
1. Submit jobs with payload and metadata (priority, delay, retries)
2. Queue and process jobs asynchronously using worker threads
3. Support multiple scheduling strategies:
   - FIFO (First In First Out)
   - Priority-based processing
   - Delayed/Scheduled processing
4. Track job status (PENDING, RUNNING, COMPLETED, FAILED)
5. Handle job failures with retry and exponential backoff
6. Provide job result retrieval and status tracking
7. Support job cancellation

NON-FUNCTIONAL REQUIREMENTS:
1. Scalability: Handle thousands of jobs per second
2. Reliability: Jobs shouldn't be lost, persistent storage
3. Fault Tolerance: Handle worker failures gracefully
4. Extensibility: Easy to add new scheduling strategies
5. Performance: Efficient job queuing and processing
6. Thread Safety: Safe concurrent access
7. Observability: Track processing metrics

DESIGN PATTERNS TO USE:
- Strategy Pattern (for different scheduling algorithms)
- Factory Pattern (for creating job processors)
- Observer Pattern (for job status notifications)
- Command Pattern (jobs as executable commands)

INTERVIEW FLOW:
Step 1: Clarify Requirements ✓
Step 2: Design Core Classes & Interfaces ✓
Step 3: Implement FIFO Job Processor (Next)
Step 4: Implement Priority Job Processor
Step 5: Implement Delayed Job Processor
Step 6: Add Factory Pattern
Step 7: Write Test Cases and Demo
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue, PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Thread, Event, Lock
import time
import uuid
import json
import logging
import random
import traceback


# ==================== ENUMS ====================

class ProcessingStrategy(Enum):
    """Enum for different job processing strategies"""
    FIFO = "fifo"
    PRIORITY = "priority"
    DELAYED = "delayed"


class JobStatus(Enum):
    """Enum for job processing states"""
    PENDING = "pending"        # Submitted but not started
    RUNNING = "running"        # Currently being processed
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Failed with error
    CANCELLED = "cancelled"    # Cancelled by user
    RETRYING = "retrying"      # Failed but retrying


# ==================== DATA MODELS ====================

@dataclass
class Job:
    """Represents a job to be processed"""
    payload: Dict[str, Any]                    # Job data/parameters
    priority: int = 3                          # Priority (1=highest, 5=lowest)
    max_retries: int = 3                       # Maximum retry attempts
    delay_seconds: float = 0.0                 # Delay before processing
    timeout_seconds: Optional[float] = None     # Job timeout
    callback: Optional[Callable] = None         # Completion callback
    
    # Internal fields
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = field(default=None, init=False)
    retry_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Calculate scheduled execution time"""
        if self.delay_seconds > 0:
            self.scheduled_at = self.created_at + timedelta(seconds=self.delay_seconds)
        else:
            self.scheduled_at = self.created_at
    
    def __lt__(self, other):
        """For priority queue ordering: priority first, then timestamp"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.scheduled_at < other.scheduled_at


@dataclass
class JobResult:
    """Represents the result of job processing"""
    job_id: str
    status: JobStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Calculate processing time if job is completed"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class JobConfig:
    """Configuration for job processors"""
    max_workers: int = 4
    max_queue_size: int = 1000
    default_priority: int = 3
    max_retries: int = 3
    retry_delay_base: float = 1.0      # Base delay for exponential backoff
    retry_delay_max: float = 60.0      # Maximum retry delay
    check_interval_seconds: float = 1.0 # How often to check for ready jobs (delayed processor)
    job_timeout_seconds: float = 300.0  # Default job timeout


# ==================== JOB PROCESSOR INTERFACE ====================

class JobProcessor(ABC):
    """Abstract base class for all job processing strategies"""
    
    def __init__(self, config: JobConfig):
        self.config = config
        self.jobs: Dict[str, Job] = {}
        self.job_results: Dict[str, JobResult] = {}
        self.is_running = False
        self.lock = Lock()  # For thread-safe operations
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
    
    @abstractmethod
    def submit_job(self, job: Job) -> str:
        """Submit a job for processing. Returns job_id"""
        pass
    
    @abstractmethod
    def start_processing(self) -> None:
        """Start job processing workers"""
        pass
    
    @abstractmethod
    def stop_processing(self) -> None:
        """Stop job processing and cleanup"""
        pass
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of a job"""
        with self.lock:
            if job_id in self.job_results:
                return self.job_results[job_id].status
            elif job_id in self.jobs:
                return JobStatus.PENDING
            return None
    
    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get job result (if completed)"""
        with self.lock:
            return self.job_results.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        with self.lock:
            if job_id in self.jobs and job_id not in self.job_results:
                # Mark as cancelled
                result = JobResult(
                    job_id=job_id,
                    status=JobStatus.CANCELLED,
                    completed_at=datetime.now()
                )
                self.job_results[job_id] = result
                return True
            return False
    
    def get_queue_size(self) -> int:
        """Get approximate queue size"""
        return len(self.jobs) - len(self.job_results)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        with self.lock:
            total_jobs = len(self.job_results)
            completed = sum(1 for r in self.job_results.values() if r.status == JobStatus.COMPLETED)
            failed = sum(1 for r in self.job_results.values() if r.status == JobStatus.FAILED)
            
            return {
                'total_jobs_processed': total_jobs,
                'completed_jobs': completed,
                'failed_jobs': failed,
                'success_rate': completed / total_jobs if total_jobs > 0 else 0,
                'queue_size': self.get_queue_size(),
                'is_running': self.is_running
            }
    
    def _execute_job(self, job: Job) -> JobResult:
        """Execute a single job and return result"""
        job_id = job.job_id
        start_time = datetime.now()
        
        # Create initial result
        result = JobResult(
            job_id=job_id,
            status=JobStatus.RUNNING,
            started_at=start_time,
            retry_count=job.retry_count
        )
        
        # Update results with running status
        with self.lock:
            self.job_results[job_id] = result
        
        try:
            self.logger.info(f"Processing job {job_id} with payload: {job.payload}")
            
            # Simulate job processing based on payload
            job_result = self._process_job_payload(job.payload)
            
            # Job completed successfully
            result.status = JobStatus.COMPLETED
            result.result = job_result
            result.completed_at = datetime.now()
            
            self.logger.info(f"Job {job_id} completed successfully in {result.processing_time_seconds:.2f}s")
            
            # Call completion callback if provided
            if job.callback:
                try:
                    job.callback(result)
                except Exception as e:
                    self.logger.warning(f"Callback failed for job {job_id}: {e}")
        
        except Exception as e:
            # Job failed
            error_msg = f"{type(e).__name__}: {str(e)}"
            result.status = JobStatus.FAILED
            result.error = error_msg
            result.completed_at = datetime.now()
            
            self.logger.error(f"Job {job_id} failed: {error_msg}")
            self.logger.debug(f"Job {job_id} traceback: {traceback.format_exc()}")
        
        # Update final result
        with self.lock:
            self.job_results[job_id] = result
        
        return result
    
    def _process_job_payload(self, payload: Dict[str, Any]) -> Any:
        """Process job payload - override for custom job logic"""
        # Default implementation simulates different types of jobs
        task_type = payload.get('task', 'default')
        
        if task_type == 'calculate':
            # Simulate CPU-intensive calculation
            data = payload.get('data', [1, 2, 3, 4, 5])
            time.sleep(0.1)  # Simulate processing time
            return {'sum': sum(data), 'avg': sum(data) / len(data)}
        
        elif task_type == 'fetch_data':
            # Simulate I/O operation
            url = payload.get('url', 'https://api.example.com/data')
            time.sleep(0.5)  # Simulate network delay
            return {'url': url, 'data': f'fetched_data_from_{url}', 'timestamp': datetime.now().isoformat()}
        
        elif task_type == 'process_file':
            # Simulate file processing
            filename = payload.get('filename', 'data.txt')
            time.sleep(0.2)  # Simulate file I/O
            return {'filename': filename, 'lines_processed': 1000, 'status': 'processed'}
        
        elif task_type == 'send_email':
            # Simulate sending email
            recipient = payload.get('recipient', 'user@example.com')
            subject = payload.get('subject', 'Test Email')
            time.sleep(0.3)  # Simulate email sending
            return {'recipient': recipient, 'subject': subject, 'sent_at': datetime.now().isoformat()}
        
        elif task_type == 'error_simulation':
            # Simulate job that might fail
            if random.random() < 0.3:  # 30% chance of failure
                raise Exception("Simulated job failure")
            return {'message': 'Job completed despite risk of failure'}
        
        else:
            # Default processing
            time.sleep(0.1)
            return {'message': f'Processed job with payload: {payload}'}
    
    def _should_retry_job(self, job: Job, result: JobResult) -> bool:
        """Determine if a failed job should be retried"""
        return (result.status == JobStatus.FAILED and 
                job.retry_count < job.max_retries)
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay for retries"""
        delay = self.config.retry_delay_base * (2 ** retry_count)
        return min(delay, self.config.retry_delay_max)


# ==================== FIFO JOB PROCESSOR ====================

class FIFOJobProcessor(JobProcessor):
    """First-In-First-Out job processing strategy"""
    
    def __init__(self, config: JobConfig):
        super().__init__(config)
        self.job_queue: Queue[Job] = Queue(maxsize=config.max_queue_size)
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.workers_shutdown_event = Event()
    
    def submit_job(self, job: Job) -> str:
        """Submit job to FIFO queue"""
        if self.job_queue.full():
            raise Exception(f"Job queue is full (max size: {self.config.max_queue_size})")
        
        with self.lock:
            self.jobs[job.job_id] = job
        
        self.job_queue.put(job)
        self.logger.info(f"Job {job.job_id} submitted to FIFO queue")
        return job.job_id
    
    def start_processing(self) -> None:
        """Start worker threads for job processing"""
        if self.is_running:
            self.logger.warning("Job processing is already running")
            return
        
        self.is_running = True
        self.workers_shutdown_event.clear()
        
        # Start worker pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="FIFOWorker"
        )
        
        # Submit worker tasks
        for i in range(self.config.max_workers):
            self.worker_pool.submit(self._worker_loop)
        
        self.logger.info(f"Started FIFO job processing with {self.config.max_workers} workers")
    
    def stop_processing(self) -> None:
        """Stop job processing gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping FIFO job processing...")
        self.is_running = False
        self.workers_shutdown_event.set()
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
        
        self.logger.info("FIFO job processing stopped")
    
    def _worker_loop(self) -> None:
        """Worker thread main loop"""
        worker_thread_name = Thread.current_thread().name
        self.logger.info(f"Worker {worker_thread_name} started")
        
        while not self.workers_shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                job = self.job_queue.get(timeout=1.0)
                
                # Check if job was cancelled
                if job.job_id in self.job_results:
                    continue
                
                # Process the job
                result = self._execute_job(job)
                
                # Handle retries for failed jobs
                if self._should_retry_job(job, result):
                    self._schedule_retry(job)
                
                self.job_queue.task_done()
                
            except Empty:
                # No job available, continue checking
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_thread_name} error: {e}")
        
        self.logger.info(f"Worker {worker_thread_name} stopped")
    
    def _schedule_retry(self, job: Job) -> None:
        """Schedule job retry with exponential backoff"""
        job.retry_count += 1
        retry_delay = self._calculate_retry_delay(job.retry_count)
        
        self.logger.info(f"Scheduling retry {job.retry_count}/{job.max_retries} for job {job.job_id} after {retry_delay:.2f}s")
        
        # Update job status to retrying
        with self.lock:
            if job.job_id in self.job_results:
                self.job_results[job.job_id].status = JobStatus.RETRYING
        
        # Schedule retry after delay
        def retry_job():
            time.sleep(retry_delay)
            if not self.workers_shutdown_event.is_set():
                self.job_queue.put(job)
        
        retry_thread = Thread(target=retry_job, name=f"RetryScheduler-{job.job_id}")
        retry_thread.daemon = True
        retry_thread.start()


# ==================== PRIORITY JOB PROCESSOR ====================

class PriorityJobProcessor(JobProcessor):
    """Priority-based job processing strategy"""
    
    def __init__(self, config: JobConfig):
        super().__init__(config)
        self.priority_queue: PriorityQueue[Job] = PriorityQueue(maxsize=config.max_queue_size)
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.workers_shutdown_event = Event()
    
    def submit_job(self, job: Job) -> str:
        """Submit job to priority queue"""
        if self.priority_queue.full():
            raise Exception(f"Job queue is full (max size: {self.config.max_queue_size})")
        
        # Use default priority if not specified
        if job.priority is None:
            job.priority = self.config.default_priority
        
        with self.lock:
            self.jobs[job.job_id] = job
        
        self.priority_queue.put(job)
        self.logger.info(f"Job {job.job_id} submitted to priority queue (priority: {job.priority})")
        return job.job_id
    
    def start_processing(self) -> None:
        """Start worker threads for priority-based processing"""
        if self.is_running:
            self.logger.warning("Job processing is already running")
            return
        
        self.is_running = True
        self.workers_shutdown_event.clear()
        
        # Start worker pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="PriorityWorker"
        )
        
        # Submit worker tasks
        for i in range(self.config.max_workers):
            self.worker_pool.submit(self._worker_loop)
        
        self.logger.info(f"Started priority job processing with {self.config.max_workers} workers")
    
    def stop_processing(self) -> None:
        """Stop job processing gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping priority job processing...")
        self.is_running = False
        self.workers_shutdown_event.set()
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
        
        self.logger.info("Priority job processing stopped")
    
    def _worker_loop(self) -> None:
        """Worker thread main loop for priority processing"""
        worker_thread_name = Thread.current_thread().name
        self.logger.info(f"Worker {worker_thread_name} started")
        
        while not self.workers_shutdown_event.is_set():
            try:
                # Get highest priority job
                job = self.priority_queue.get(timeout=1.0)
                
                # Check if job was cancelled
                if job.job_id in self.job_results:
                    continue
                
                self.logger.info(f"Processing priority job {job.job_id} (priority: {job.priority})")
                
                # Process the job
                result = self._execute_job(job)
                
                # Handle retries for failed jobs
                if self._should_retry_job(job, result):
                    self._schedule_retry(job)
                
                self.priority_queue.task_done()
                
            except Empty:
                # No job available, continue checking
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_thread_name} error: {e}")
        
        self.logger.info(f"Worker {worker_thread_name} stopped")
    
    def _schedule_retry(self, job: Job) -> None:
        """Schedule job retry with exponential backoff"""
        job.retry_count += 1
        retry_delay = self._calculate_retry_delay(job.retry_count)
        
        self.logger.info(f"Scheduling priority retry {job.retry_count}/{job.max_retries} for job {job.job_id} after {retry_delay:.2f}s")
        
        # Update job status to retrying
        with self.lock:
            if job.job_id in self.job_results:
                self.job_results[job.job_id].status = JobStatus.RETRYING
        
        # Schedule retry after delay
        def retry_job():
            time.sleep(retry_delay)
            if not self.workers_shutdown_event.is_set():
                self.priority_queue.put(job)
        
        retry_thread = Thread(target=retry_job, name=f"PriorityRetryScheduler-{job.job_id}")
        retry_thread.daemon = True
        retry_thread.start()


# ==================== DELAYED JOB PROCESSOR ====================

class DelayedJobProcessor(JobProcessor):
    """Time-based/delayed job processing strategy"""
    
    def __init__(self, config: JobConfig):
        super().__init__(config)
        self.scheduled_jobs: PriorityQueue[tuple] = PriorityQueue()  # (scheduled_time, job)
        self.ready_jobs: Queue[Job] = Queue()
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.scheduler_thread: Optional[Thread] = None
        self.workers_shutdown_event = Event()
    
    def submit_job(self, job: Job) -> str:
        """Submit job for delayed processing"""
        with self.lock:
            self.jobs[job.job_id] = job
        
        # Calculate when job should be processed
        if job.scheduled_at is None:
            job.scheduled_at = datetime.now() + timedelta(seconds=job.delay_seconds)
        
        # Add to scheduled queue with timestamp for ordering
        scheduled_time_timestamp = job.scheduled_at.timestamp()
        self.scheduled_jobs.put((scheduled_time_timestamp, job))
        
        delay_info = f"immediately" if job.delay_seconds == 0 else f"in {job.delay_seconds}s"
        self.logger.info(f"Job {job.job_id} scheduled for processing {delay_info}")
        return job.job_id
    
    def start_processing(self) -> None:
        """Start delayed job processing"""
        if self.is_running:
            self.logger.warning("Job processing is already running")
            return
        
        self.is_running = True
        self.workers_shutdown_event.clear()
        
        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._scheduler_loop, name="DelayedJobScheduler")
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Start worker pool
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="DelayedWorker"
        )
        
        # Submit worker tasks
        for i in range(self.config.max_workers):
            self.worker_pool.submit(self._worker_loop)
        
        self.logger.info(f"Started delayed job processing with {self.config.max_workers} workers")
    
    def stop_processing(self) -> None:
        """Stop delayed job processing gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping delayed job processing...")
        self.is_running = False
        self.workers_shutdown_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2.0)
        
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            self.worker_pool = None
        
        self.logger.info("Delayed job processing stopped")
    
    def _scheduler_loop(self) -> None:
        """Scheduler thread that moves ready jobs to processing queue"""
        self.logger.info("Delayed job scheduler started")
        
        while not self.workers_shutdown_event.is_set():
            try:
                current_time = datetime.now().timestamp()
                
                # Check for ready jobs
                ready_jobs_to_process = []
                
                # Get jobs that are ready (non-blocking peek)
                temp_jobs = []
                while not self.scheduled_jobs.empty():
                    try:
                        scheduled_time, job = self.scheduled_jobs.get_nowait()
                        if scheduled_time <= current_time:
                            ready_jobs_to_process.append(job)
                        else:
                            temp_jobs.append((scheduled_time, job))
                    except:
                        break
                
                # Put non-ready jobs back
                for scheduled_time, job in temp_jobs:
                    self.scheduled_jobs.put((scheduled_time, job))
                
                # Move ready jobs to processing queue
                for job in ready_jobs_to_process:
                    # Check if job was cancelled
                    if job.job_id not in self.job_results:
                        self.ready_jobs.put(job)
                        self.logger.info(f"Job {job.job_id} is now ready for processing")
                
                # Sleep before next check
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                time.sleep(1.0)
        
        self.logger.info("Delayed job scheduler stopped")
    
    def _worker_loop(self) -> None:
        """Worker thread main loop for delayed jobs"""
        worker_thread_name = Thread.current_thread().name
        self.logger.info(f"Worker {worker_thread_name} started")
        
        while not self.workers_shutdown_event.is_set():
            try:
                # Get ready job
                job = self.ready_jobs.get(timeout=1.0)
                
                # Check if job was cancelled
                if job.job_id in self.job_results:
                    continue
                
                self.logger.info(f"Processing delayed job {job.job_id}")
                
                # Process the job
                result = self._execute_job(job)
                
                # Handle retries for failed jobs
                if self._should_retry_job(job, result):
                    self._schedule_retry(job)
                
                self.ready_jobs.task_done()
                
            except Empty:
                # No job available, continue checking
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_thread_name} error: {e}")
        
        self.logger.info(f"Worker {worker_thread_name} stopped")
    
    def _schedule_retry(self, job: Job) -> None:
        """Schedule job retry with exponential backoff"""
        job.retry_count += 1
        retry_delay = self._calculate_retry_delay(job.retry_count)
        
        # Update scheduled time for retry
        job.scheduled_at = datetime.now() + timedelta(seconds=retry_delay)
        
        self.logger.info(f"Scheduling delayed retry {job.retry_count}/{job.max_retries} for job {job.job_id} at {job.scheduled_at}")
        
        # Update job status to retrying
        with self.lock:
            if job.job_id in self.job_results:
                self.job_results[job.job_id].status = JobStatus.RETRYING
        
        # Add back to scheduled queue
        scheduled_time_timestamp = job.scheduled_at.timestamp()
        self.scheduled_jobs.put((scheduled_time_timestamp, job))


# ==================== FACTORY ====================

class JobProcessorFactory:
    """Factory for creating job processors"""
    
    @staticmethod
    def create(strategy: ProcessingStrategy, **config_kwargs) -> JobProcessor:
        """Create a job processor with specified strategy and configuration"""
        
        # Create configuration
        config = JobConfig(**config_kwargs)
        
        # Create appropriate processor
        if strategy == ProcessingStrategy.FIFO:
            return FIFOJobProcessor(config)
        elif strategy == ProcessingStrategy.PRIORITY:
            return PriorityJobProcessor(config)
        elif strategy == ProcessingStrategy.DELAYED:
            return DelayedJobProcessor(config)
        else:
            raise ValueError(f"Unknown processing strategy: {strategy}")
    
    @staticmethod
    def create_from_string(strategy_name: str, **config_kwargs) -> JobProcessor:
        """Create job processor from string name"""
        try:
            strategy = ProcessingStrategy(strategy_name.lower())
            return JobProcessorFactory.create(strategy, **config_kwargs)
        except ValueError:
            available = [s.value for s in ProcessingStrategy]
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")


# ==================== DEMO AND TESTING ====================

def create_sample_jobs() -> List[Job]:
    """Create sample jobs for testing"""
    jobs = []
    
    # High priority calculation job
    jobs.append(Job(
        payload={'task': 'calculate', 'data': [1, 2, 3, 4, 5]},
        priority=1,
        max_retries=2
    ))
    
    # Medium priority data fetching
    jobs.append(Job(
        payload={'task': 'fetch_data', 'url': 'https://api.example.com/users'},
        priority=3,
        max_retries=3
    ))
    
    # Low priority file processing with delay
    jobs.append(Job(
        payload={'task': 'process_file', 'filename': 'large_report.csv'},
        priority=5,
        delay_seconds=2.0,
        max_retries=1
    ))
    
    # Email notification job
    jobs.append(Job(
        payload={'task': 'send_email', 'recipient': 'admin@company.com', 'subject': 'Daily Report'},
        priority=2,
        max_retries=5
    ))
    
    # Job that might fail (for retry testing)
    jobs.append(Job(
        payload={'task': 'error_simulation'},
        priority=3,
        max_retries=3
    ))
    
    return jobs


def demo_fifo_processor():
    """Demonstrate FIFO job processing"""
    print("\n" + "="*60)
    print("🔄 FIFO JOB PROCESSOR DEMO")
    print("="*60)
    
    # Create FIFO processor
    config = JobConfig(max_workers=2, max_queue_size=10)
    processor = JobProcessorFactory.create(ProcessingStrategy.FIFO, **config.__dict__)
    
    try:
        # Submit jobs
        jobs = create_sample_jobs()[:3]  # Use first 3 jobs
        job_ids = []
        
        print(f"\n📝 Submitting {len(jobs)} jobs to FIFO processor...")
        for i, job in enumerate(jobs):
            job_id = processor.submit_job(job)
            job_ids.append(job_id)
            print(f"   Job {i+1}: {job_id} - {job.payload['task']}")
        
        # Start processing
        print("\n🚀 Starting FIFO job processing...")
        processor.start_processing()
        
        # Monitor progress
        print("\n📊 Monitoring job progress...")
        completed_jobs = 0
        while completed_jobs < len(job_ids):
            time.sleep(1)
            completed_jobs = 0
            
            for job_id in job_ids:
                status = processor.get_job_status(job_id)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    completed_jobs += 1
            
            metrics = processor.get_metrics()
            print(f"   Progress: {completed_jobs}/{len(job_ids)} jobs completed, Queue: {metrics['queue_size']}")
        
        # Show results
        print("\n✅ All jobs completed! Results:")
        for job_id in job_ids:
            result = processor.get_job_result(job_id)
            if result:
                status_emoji = "✅" if result.status == JobStatus.COMPLETED else "❌"
                processing_time = result.processing_time_seconds or 0
                print(f"   {status_emoji} {job_id}: {result.status.value} ({processing_time:.2f}s)")
                if result.result:
                    print(f"      Result: {result.result}")
        
        # Show final metrics
        final_metrics = processor.get_metrics()
        print(f"\n📈 Final metrics: {final_metrics}")
    
    finally:
        processor.stop_processing()


def demo_priority_processor():
    """Demonstrate priority-based job processing"""
    print("\n" + "="*60)
    print("🎯 PRIORITY JOB PROCESSOR DEMO")
    print("="*60)
    
    # Create priority processor
    config = JobConfig(max_workers=2, max_queue_size=10)
    processor = JobProcessorFactory.create(ProcessingStrategy.PRIORITY, **config.__dict__)
    
    try:
        # Submit jobs (they should be processed in priority order)
        jobs = create_sample_jobs()
        job_ids = []
        
        print(f"\n📝 Submitting {len(jobs)} jobs to priority processor...")
        for i, job in enumerate(jobs):
            job_id = processor.submit_job(job)
            job_ids.append(job_id)
            print(f"   Job {i+1}: {job_id} - {job.payload['task']} (Priority: {job.priority})")
        
        # Start processing
        print("\n🚀 Starting priority job processing...")
        print("   (Higher priority jobs should be processed first)")
        processor.start_processing()
        
        # Monitor progress
        print("\n📊 Monitoring job progress...")
        completed_jobs = 0
        while completed_jobs < len(job_ids):
            time.sleep(1)
            completed_jobs = 0
            
            for job_id in job_ids:
                status = processor.get_job_status(job_id)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    completed_jobs += 1
            
            metrics = processor.get_metrics()
            print(f"   Progress: {completed_jobs}/{len(job_ids)} jobs completed")
        
        # Show results
        print("\n✅ All jobs completed! Results:")
        for job_id in job_ids:
            result = processor.get_job_result(job_id)
            if result:
                status_emoji = "✅" if result.status == JobStatus.COMPLETED else "❌"
                processing_time = result.processing_time_seconds or 0
                print(f"   {status_emoji} {job_id}: {result.status.value} ({processing_time:.2f}s)")
        
        # Show final metrics
        final_metrics = processor.get_metrics()
        print(f"\n📈 Final metrics: {final_metrics}")
    
    finally:
        processor.stop_processing()


def demo_delayed_processor():
    """Demonstrate delayed job processing"""
    print("\n" + "="*60)
    print("⏰ DELAYED JOB PROCESSOR DEMO")
    print("="*60)
    
    # Create delayed processor
    config = JobConfig(max_workers=2, check_interval_seconds=0.5)
    processor = JobProcessorFactory.create(ProcessingStrategy.DELAYED, **config.__dict__)
    
    try:
        # Create jobs with various delays
        immediate_job = Job(
            payload={'task': 'calculate', 'data': [10, 20, 30]},
            delay_seconds=0
        )
        
        delayed_job = Job(
            payload={'task': 'fetch_data', 'url': 'https://api.example.com/delayed'},
            delay_seconds=2.0
        )
        
        future_job = Job(
            payload={'task': 'send_email', 'recipient': 'test@example.com'},
            delay_seconds=4.0
        )
        
        jobs = [immediate_job, delayed_job, future_job]
        job_ids = []
        
        print(f"\n📝 Submitting {len(jobs)} jobs with delays...")
        for i, job in enumerate(jobs):
            job_id = processor.submit_job(job)
            job_ids.append(job_id)
            delay_info = "immediately" if job.delay_seconds == 0 else f"after {job.delay_seconds}s"
            print(f"   Job {i+1}: {job_id} - {job.payload['task']} (Process {delay_info})")
        
        # Start processing
        print("\n🚀 Starting delayed job processing...")
        processor.start_processing()
        
        # Monitor progress with timestamps
        print("\n📊 Monitoring job progress...")
        start_time = datetime.now()
        completed_jobs = 0
        
        while completed_jobs < len(job_ids):
            time.sleep(0.5)
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            
            completed_jobs = 0
            for job_id in job_ids:
                status = processor.get_job_status(job_id)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    completed_jobs += 1
            
            print(f"   T+{elapsed:.1f}s: {completed_jobs}/{len(job_ids)} jobs completed")
        
        # Show results
        print("\n✅ All jobs completed! Results:")
        for job_id in job_ids:
            result = processor.get_job_result(job_id)
            if result:
                status_emoji = "✅" if result.status == JobStatus.COMPLETED else "❌"
                processing_time = result.processing_time_seconds or 0
                print(f"   {status_emoji} {job_id}: {result.status.value} ({processing_time:.2f}s)")
        
        # Show final metrics
        final_metrics = processor.get_metrics()
        print(f"\n📈 Final metrics: {final_metrics}")
    
    finally:
        processor.stop_processing()


def demo_job_cancellation():
    """Demonstrate job cancellation"""
    print("\n" + "="*60)
    print("❌ JOB CANCELLATION DEMO")
    print("="*60)
    
    # Create processor with delayed jobs
    config = JobConfig(max_workers=1, check_interval_seconds=0.5)
    processor = JobProcessorFactory.create(ProcessingStrategy.DELAYED, **config.__dict__)
    
    try:
        # Create jobs with delays
        job1 = Job(payload={'task': 'calculate', 'data': [1, 2, 3]}, delay_seconds=1.0)
        job2 = Job(payload={'task': 'calculate', 'data': [4, 5, 6]}, delay_seconds=2.0)
        job3 = Job(payload={'task': 'calculate', 'data': [7, 8, 9]}, delay_seconds=3.0)
        
        print("\n📝 Submitting jobs...")
        job_id1 = processor.submit_job(job1)
        job_id2 = processor.submit_job(job2)
        job_id3 = processor.submit_job(job3)
        
        print(f"   Job 1: {job_id1} (1s delay)")
        print(f"   Job 2: {job_id2} (2s delay)")
        print(f"   Job 3: {job_id3} (3s delay)")
        
        # Start processing
        print("\n🚀 Starting processing...")
        processor.start_processing()
        
        # Cancel middle job after 1.5 seconds
        time.sleep(1.5)
        success = processor.cancel_job(job_id2)
        print(f"\n❌ Cancelled job 2: {success}")
        
        # Wait for remaining jobs
        time.sleep(3.0)
        
        # Show results
        print("\n📊 Final results:")
        for job_id, name in [(job_id1, "Job 1"), (job_id2, "Job 2 (cancelled)"), (job_id3, "Job 3")]:
            status = processor.get_job_status(job_id)
            result = processor.get_job_result(job_id)
            status_emoji = "✅" if status == JobStatus.COMPLETED else "❌" if status == JobStatus.CANCELLED else "⏳"
            print(f"   {status_emoji} {name}: {status.value}")
    
    finally:
        processor.stop_processing()


def run_all_demos():
    """Run all job processor demos"""
    print("🎯 DISTRIBUTED JOB PROCESSING SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    try:
        # Demo each processing strategy
        demo_fifo_processor()
        demo_priority_processor()
        demo_delayed_processor()
        demo_job_cancellation()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ FIFO job processing (fair, ordered)")
        print("   ✅ Priority-based processing (important jobs first)")
        print("   ✅ Delayed/scheduled job processing")
        print("   ✅ Job status tracking and monitoring")
        print("   ✅ Automatic retry with exponential backoff")
        print("   ✅ Job cancellation")
        print("   ✅ Thread-safe concurrent processing")
        print("   ✅ Configurable worker pools")
        print("   ✅ Comprehensive metrics")
        
        print("\n🚀 Production Considerations:")
        print("   • Use Redis for distributed job queues")
        print("   • Add job persistence for reliability")
        print("   • Implement worker health monitoring")
        print("   • Add circuit breakers for fault tolerance")
        print("   • Monitor queue depth and processing rates")
        print("   • Consider job dependencies and workflows")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the comprehensive demo
    run_all_demos()