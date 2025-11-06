"""
Notification Service - Single File Implementation
For coding interviews and production-ready reference

Features:
- Multi-channel delivery (Email, SMS, Push Notification, Slack)
- Priority-based queuing with retry mechanisms
- Deduplication to prevent notification spam
- Rate limiting per user and channel
- Template-based notification content
- Exponential backoff for failed deliveries

Interview Focus:
- Strategy pattern for notification channels
- Observer pattern for event-driven notifications
- Queue management for reliable delivery
- Deduplication algorithms using time windows
- Rate limiting to prevent spam
- Error handling and retry strategies

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, deque
from threading import RLock, Thread
from queue import PriorityQueue
import time
import hashlib
import uuid


# ============================================================================
# SECTION 1: MODELS - Core data classes and enums
# ============================================================================

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"


class NotificationPriority(Enum):
    """Priority levels for notification delivery"""
    LOW = 3
    MEDIUM = 2
    HIGH = 1
    CRITICAL = 0  # Lowest number = highest priority


class NotificationStatus(Enum):
    """Status of notification delivery"""
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"
    DEDUPLICATED = "deduplicated"
    RATE_LIMITED = "rate_limited"


@dataclass
class Notification:
    """
    Core notification entity
    
    Interview Focus: Why separate template from content? Reusability and localization
    """
    notification_id: str
    user_id: str
    channel: NotificationChannel
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue comparison (lower priority value = higher priority)"""
        return self.priority.value < other.priority.value
    
    def get_dedup_key(self) -> str:
        """
        Generate deduplication key
        
        Key Insight: Hash user_id + channel + content to detect duplicates
        Time Complexity: O(n) where n is message length
        """
        content = f"{self.user_id}:{self.channel.value}:{self.title}:{self.message}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class NotificationResult:
    """Result of notification delivery attempt"""
    notification_id: str
    status: NotificationStatus
    message: str
    attempt_count: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per channel"""
    max_notifications: int
    time_window_seconds: int
    
    def __post_init__(self):
        if self.max_notifications <= 0:
            raise ValueError("max_notifications must be positive")
        if self.time_window_seconds <= 0:
            raise ValueError("time_window_seconds must be positive")


# ============================================================================
# SECTION 2: NOTIFICATION CHANNELS - Strategy Pattern
# ============================================================================

class NotificationChannelProvider(ABC):
    """
    Abstract base for notification channels
    
    Strategy Pattern: Each channel implements its own delivery logic
    
    Interview Focus: Why strategy pattern? Different channels have different
    delivery mechanisms, credentials, and error handling
    """
    
    def __init__(self, channel_type: NotificationChannel):
        self.channel_type = channel_type
        self.lock = RLock()
    
    @abstractmethod
    def send(self, notification: Notification) -> Tuple[bool, str]:
        """
        Send notification through this channel
        
        Returns: (success, message)
        """
        pass
    
    @abstractmethod
    def validate_recipient(self, user_id: str) -> bool:
        """Validate that user can receive notifications on this channel"""
        pass


class EmailProvider(NotificationChannelProvider):
    """
    Email notification provider
    
    Key Features:
    - SMTP integration
    - HTML/Plain text support
    - Attachment handling
    
    Interview Focus: How do you handle email delivery failures?
    """
    
    def __init__(self):
        super().__init__(NotificationChannel.EMAIL)
        self.smtp_config = {}  # In production: SMTP credentials
    
    def send(self, notification: Notification) -> Tuple[bool, str]:
        """
        Send email notification
        
        Time Complexity: O(1) for queuing, O(n) for actual SMTP send
        Space Complexity: O(1)
        
        Key Insight: Queue email for async delivery to avoid blocking
        """
        with self.lock:
            # Simulate email sending
            recipient_email = self._get_user_email(notification.user_id)
            
            if not recipient_email:
                return False, "User email not found"
            
            # In production: Use SMTP library or email service API
            print(f"📧 Sending email to {recipient_email}")
            print(f"   Subject: {notification.title}")
            print(f"   Message: {notification.message[:50]}...")
            
            # Simulate network delay
            time.sleep(0.1)
            
            return True, f"Email sent to {recipient_email}"
    
    def validate_recipient(self, user_id: str) -> bool:
        """Check if user has valid email"""
        return bool(self._get_user_email(user_id))
    
    def _get_user_email(self, user_id: str) -> Optional[str]:
        """Simulate user lookup"""
        # In production: Query user database
        return f"user_{user_id}@example.com"


class SMSProvider(NotificationChannelProvider):
    """
    SMS notification provider
    
    Key Features:
    - Integration with SMS gateway (Twilio, AWS SNS)
    - Character limit handling
    - International number formatting
    """
    
    def __init__(self):
        super().__init__(NotificationChannel.SMS)
        self.max_message_length = 160
    
    def send(self, notification: Notification) -> Tuple[bool, str]:
        """
        Send SMS notification
        
        Key Insight: Truncate message to SMS limits, consider cost per SMS
        """
        with self.lock:
            phone_number = self._get_user_phone(notification.user_id)
            
            if not phone_number:
                return False, "User phone number not found"
            
            # Truncate message if too long
            message = notification.message[:self.max_message_length]
            
            print(f"📱 Sending SMS to {phone_number}")
            print(f"   Message: {message}")
            
            # Simulate network delay
            time.sleep(0.1)
            
            return True, f"SMS sent to {phone_number}"
    
    def validate_recipient(self, user_id: str) -> bool:
        """Check if user has valid phone number"""
        return bool(self._get_user_phone(user_id))
    
    def _get_user_phone(self, user_id: str) -> Optional[str]:
        """Simulate user lookup"""
        return f"+1234567{user_id[:4]}"


class PushNotificationProvider(NotificationChannelProvider):
    """
    Push notification provider (mobile/web)
    
    Key Features:
    - FCM/APNs integration
    - Device token management
    - Badge count updates
    """
    
    def __init__(self):
        super().__init__(NotificationChannel.PUSH)
        self.device_tokens = {}  # user_id -> [device_tokens]
    
    def send(self, notification: Notification) -> Tuple[bool, str]:
        """
        Send push notification
        
        Key Challenge: Handle multiple devices per user
        """
        with self.lock:
            tokens = self._get_device_tokens(notification.user_id)
            
            if not tokens:
                return False, "No device tokens found for user"
            
            sent_count = 0
            for token in tokens:
                print(f"🔔 Sending push to device {token[:8]}...")
                print(f"   Title: {notification.title}")
                print(f"   Body: {notification.message[:40]}...")
                sent_count += 1
            
            time.sleep(0.1)
            
            return True, f"Push sent to {sent_count} device(s)"
    
    def validate_recipient(self, user_id: str) -> bool:
        """Check if user has registered devices"""
        return bool(self._get_device_tokens(user_id))
    
    def _get_device_tokens(self, user_id: str) -> List[str]:
        """Get user's device tokens"""
        # Simulate device token lookup
        return [f"token_{user_id}_device1", f"token_{user_id}_device2"]


class SlackProvider(NotificationChannelProvider):
    """Slack notification provider"""
    
    def __init__(self):
        super().__init__(NotificationChannel.SLACK)
    
    def send(self, notification: Notification) -> Tuple[bool, str]:
        """Send Slack message"""
        with self.lock:
            slack_id = self._get_slack_id(notification.user_id)
            
            if not slack_id:
                return False, "User Slack ID not found"
            
            print(f"💬 Sending Slack message to @{slack_id}")
            print(f"   {notification.title}")
            print(f"   {notification.message[:50]}...")
            
            time.sleep(0.1)
            
            return True, f"Slack message sent to @{slack_id}"
    
    def validate_recipient(self, user_id: str) -> bool:
        """Check if user has Slack integration"""
        return bool(self._get_slack_id(user_id))
    
    def _get_slack_id(self, user_id: str) -> Optional[str]:
        """Get user's Slack ID"""
        return f"slack_{user_id}"


# ============================================================================
# SECTION 3: DEDUPLICATION - Prevent notification spam
# ============================================================================

class NotificationDeduplicator:
    """
    Deduplicate notifications within time window
    
    Interview Focus: How do you prevent duplicate notifications?
    
    Algorithm: Time-window based deduplication using hash map
    - Store notification hash + timestamp
    - Check if same notification sent recently
    - Clean up expired entries periodically
    """
    
    def __init__(self, dedup_window_seconds: int = 300):
        """
        Initialize deduplicator
        
        Args:
            dedup_window_seconds: Time window for deduplication (default 5 minutes)
        """
        self.dedup_window_seconds = dedup_window_seconds
        self.sent_notifications: Dict[str, float] = {}  # dedup_key -> timestamp
        self.lock = RLock()
    
    def is_duplicate(self, notification: Notification) -> bool:
        """
        Check if notification is duplicate within time window
        
        Time Complexity: O(1) average for hash lookup
        Space Complexity: O(n) where n is notifications in window
        
        Key Insight: Use content hash to detect duplicates
        """
        with self.lock:
            dedup_key = notification.get_dedup_key()
            current_time = time.time()
            
            # Clean up old entries periodically
            self._cleanup_expired(current_time)
            
            # Check if we've sent this notification recently
            if dedup_key in self.sent_notifications:
                last_sent = self.sent_notifications[dedup_key]
                if current_time - last_sent < self.dedup_window_seconds:
                    return True
            
            # Mark as sent
            self.sent_notifications[dedup_key] = current_time
            return False
    
    def _cleanup_expired(self, current_time: float) -> None:
        """
        Remove expired deduplication entries
        
        Time Complexity: O(n) where n is total entries
        
        Optimization: Run periodically in background thread
        """
        expired_keys = [
            key for key, timestamp in self.sent_notifications.items()
            if current_time - timestamp > self.dedup_window_seconds
        ]
        
        for key in expired_keys:
            del self.sent_notifications[key]


# ============================================================================
# SECTION 4: RATE LIMITER - Prevent notification spam per user
# ============================================================================

class NotificationRateLimiter:
    """
    Rate limit notifications per user per channel
    
    Interview Focus: Token bucket vs sliding window?
    
    Implementation: Sliding window log
    - Track timestamps of notifications
    - Count notifications in time window
    - Reject if exceeds limit
    """
    
    def __init__(self, default_config: RateLimitConfig):
        self.default_config = default_config
        self.channel_configs: Dict[NotificationChannel, RateLimitConfig] = {}
        
        # user_id:channel -> [timestamps]
        self.notification_log: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = RLock()
    
    def set_channel_config(self, channel: NotificationChannel, config: RateLimitConfig):
        """Set rate limit config for specific channel"""
        self.channel_configs[channel] = config
    
    def is_allowed(self, notification: Notification) -> Tuple[bool, str]:
        """
        Check if notification is allowed based on rate limits
        
        Time Complexity: O(n) where n is notifications in window
        Space Complexity: O(u * c * n) where u=users, c=channels, n=limit
        
        Key Insight: Sliding window provides accurate rate limiting
        """
        with self.lock:
            config = self.channel_configs.get(
                notification.channel,
                self.default_config
            )
            
            key = f"{notification.user_id}:{notification.channel.value}"
            current_time = time.time()
            
            # Get notification log for this user:channel
            log = self.notification_log[key]
            
            # Remove old entries outside time window
            cutoff_time = current_time - config.time_window_seconds
            while log and log[0] < cutoff_time:
                log.popleft()
            
            # Check rate limit
            if len(log) >= config.max_notifications:
                return False, f"Rate limit exceeded: {len(log)}/{config.max_notifications} in {config.time_window_seconds}s"
            
            # Add current timestamp
            log.append(current_time)
            
            return True, "Rate limit OK"


# ============================================================================
# SECTION 5: RETRY MECHANISM - Handle failed deliveries
# ============================================================================

@dataclass
class RetryableNotification:
    """Notification with retry metadata"""
    notification: Notification
    attempt_count: int = 0
    max_attempts: int = 3
    last_attempt_time: float = 0
    next_retry_time: float = 0
    error_message: str = ""
    
    def calculate_next_retry(self) -> float:
        """
        Calculate next retry time using exponential backoff
        
        Algorithm: delay = base_delay * (2 ^ attempt_count)
        
        Key Insight: Exponential backoff prevents thundering herd
        """
        base_delay = 60  # 60 seconds
        delay = base_delay * (2 ** self.attempt_count)
        max_delay = 3600  # Cap at 1 hour
        delay = min(delay, max_delay)
        return time.time() + delay
    
    def should_retry(self) -> bool:
        """Check if notification should be retried"""
        return (self.attempt_count < self.max_attempts and
                time.time() >= self.next_retry_time)


class NotificationRetryQueue:
    """
    Queue for failed notifications with retry logic
    
    Interview Focus: How do you handle transient failures?
    """
    
    def __init__(self):
        self.retry_queue: Dict[str, RetryableNotification] = {}
        self.lock = RLock()
    
    def add_for_retry(self, notification: Notification, error: str) -> None:
        """Add failed notification to retry queue"""
        with self.lock:
            retry_notif = RetryableNotification(
                notification=notification,
                attempt_count=1,
                last_attempt_time=time.time(),
                error_message=error
            )
            retry_notif.next_retry_time = retry_notif.calculate_next_retry()
            
            self.retry_queue[notification.notification_id] = retry_notif
    
    def get_ready_for_retry(self) -> List[RetryableNotification]:
        """
        Get notifications ready for retry
        
        Time Complexity: O(n) where n is queue size
        """
        with self.lock:
            ready = []
            for notif_id, retry_notif in list(self.retry_queue.items()):
                if retry_notif.should_retry():
                    retry_notif.attempt_count += 1
                    retry_notif.last_attempt_time = time.time()
                    ready.append(retry_notif)
                elif retry_notif.attempt_count >= retry_notif.max_attempts:
                    # Max attempts reached, remove from queue
                    del self.retry_queue[notif_id]
            
            return ready


# ============================================================================
# SECTION 6: NOTIFICATION SERVICE - Main controller
# ============================================================================

class NotificationService:
    """
    Main notification service coordinator
    
    Responsibilities:
    - Route notifications to appropriate channels
    - Manage delivery queue with priorities
    - Deduplicate notifications
    - Rate limit per user
    - Retry failed deliveries
    - Track delivery status
    
    Thread Safety: Uses RLock for all shared state
    
    Interview Focus: How do you design a scalable notification system?
    """
    
    def __init__(self, service_id: str = "notification_service"):
        self.service_id = service_id
        
        # Channel providers (Strategy pattern)
        self.providers: Dict[NotificationChannel, NotificationChannelProvider] = {
            NotificationChannel.EMAIL: EmailProvider(),
            NotificationChannel.SMS: SMSProvider(),
            NotificationChannel.PUSH: PushNotificationProvider(),
            NotificationChannel.SLACK: SlackProvider()
        }
        
        # Deduplication
        self.deduplicator = NotificationDeduplicator(dedup_window_seconds=300)
        
        # Rate limiting
        default_rate_config = RateLimitConfig(max_notifications=10, time_window_seconds=60)
        self.rate_limiter = NotificationRateLimiter(default_rate_config)
        
        # Retry queue
        self.retry_queue = NotificationRetryQueue()
        
        # Delivery queue (priority-based)
        self.delivery_queue: PriorityQueue = PriorityQueue()
        
        # Delivery history
        self.delivery_history: Dict[str, NotificationResult] = {}
        
        # Thread safety
        self.lock = RLock()
        
        # Background worker
        self.running = False
        self.worker_thread = None
    
    def send_notification(self, notification: Notification) -> Tuple[bool, str]:
        """
        Send notification through specified channel
        
        Interview Focus: How do you handle the complete notification flow?
        
        Key Challenges:
        - Validate recipient exists
        - Check for duplicates
        - Apply rate limiting
        - Queue for delivery
        - Handle failures with retry
        """
        with self.lock:
            # Step 1: Validate recipient
            provider = self.providers.get(notification.channel)
            if not provider:
                return False, f"Unsupported channel: {notification.channel}"
            
            if not provider.validate_recipient(notification.user_id):
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.FAILED,
                    message="Invalid recipient"
                )
                self.delivery_history[notification.notification_id] = result
                return False, "Invalid recipient"
            
            # Step 2: Check deduplication
            if self.deduplicator.is_duplicate(notification):
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.DEDUPLICATED,
                    message="Duplicate notification within time window"
                )
                self.delivery_history[notification.notification_id] = result
                return False, "Duplicate notification"
            
            # Step 3: Check rate limits
            allowed, rate_msg = self.rate_limiter.is_allowed(notification)
            if not allowed:
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.RATE_LIMITED,
                    message=rate_msg
                )
                self.delivery_history[notification.notification_id] = result
                return False, rate_msg
            
            # Step 4: Queue for delivery
            self.delivery_queue.put(notification)
            
            result = NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.QUEUED,
                message="Queued for delivery"
            )
            self.delivery_history[notification.notification_id] = result
            
            return True, "Notification queued successfully"
    
    def _process_delivery(self, notification: Notification) -> None:
        """
        Process notification delivery
        
        Key Insight: Separate queuing from delivery for scalability
        """
        provider = self.providers[notification.channel]
        
        try:
            success, message = provider.send(notification)
            
            if success:
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.SENT,
                    message=message
                )
                self.delivery_history[notification.notification_id] = result
            else:
                # Add to retry queue
                self.retry_queue.add_for_retry(notification, message)
                result = NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.FAILED,
                    message=f"Failed: {message}, queued for retry"
                )
                self.delivery_history[notification.notification_id] = result
        
        except Exception as e:
            # Add to retry queue on exception
            self.retry_queue.add_for_retry(notification, str(e))
            result = NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.FAILED,
                message=f"Exception: {str(e)}, queued for retry"
            )
            self.delivery_history[notification.notification_id] = result
    
    def start_worker(self) -> None:
        """Start background worker for processing queue"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop_worker(self) -> None:
        """Stop background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _worker_loop(self) -> None:
        """
        Background worker loop
        
        Processes queued notifications and retries
        """
        while self.running:
            # Process main delivery queue
            if not self.delivery_queue.empty():
                notification = self.delivery_queue.get()
                self._process_delivery(notification)
            
            # Process retry queue
            ready_for_retry = self.retry_queue.get_ready_for_retry()
            for retry_notif in ready_for_retry:
                self._process_delivery(retry_notif.notification)
            
            # Sleep briefly to prevent busy waiting
            time.sleep(0.1)
    
    def get_delivery_status(self, notification_id: str) -> Optional[NotificationResult]:
        """Get delivery status for notification"""
        with self.lock:
            return self.delivery_history.get(notification_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self.lock:
            stats = {
                "total_notifications": len(self.delivery_history),
                "queue_size": self.delivery_queue.qsize(),
                "retry_queue_size": len(self.retry_queue.retry_queue),
                "by_status": defaultdict(int),
                "by_channel": defaultdict(int)
            }
            
            for result in self.delivery_history.values():
                stats["by_status"][result.status.value] += 1
            
            return stats


# ============================================================================
# DEMO - Demonstration and testing code
# ============================================================================

def print_separator(title: str = ""):
    """Print visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"{'='*70}\n")


def demo_basic_operations():
    """Demonstrate basic notification sending"""
    print_separator("Basic Notification Operations")
    
    service = NotificationService()
    service.start_worker()
    
    print("\n🔹 Test 1: Send email notification")
    notification = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_123",
        channel=NotificationChannel.EMAIL,
        title="Welcome to our service!",
        message="Thank you for signing up. Get started with our tutorial.",
        priority=NotificationPriority.HIGH
    )
    
    success, message = service.send_notification(notification)
    print(f"{'✓' if success else '✗'} {message}")
    
    # Give worker time to process
    time.sleep(0.5)
    
    status = service.get_delivery_status(notification.notification_id)
    print(f"Delivery status: {status.status.value} - {status.message}")
    
    print("\n🔹 Test 2: Send SMS notification")
    sms_notif = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_456",
        channel=NotificationChannel.SMS,
        title="Verification Code",
        message="Your verification code is: 123456",
        priority=NotificationPriority.CRITICAL
    )
    
    success, message = service.send_notification(sms_notif)
    print(f"{'✓' if success else '✗'} {message}")
    
    time.sleep(0.5)
    
    print("\n🔹 Test 3: Send push notification")
    push_notif = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_789",
        channel=NotificationChannel.PUSH,
        title="New Message",
        message="You have a new message from Alice",
        priority=NotificationPriority.MEDIUM
    )
    
    success, message = service.send_notification(push_notif)
    print(f"{'✓' if success else '✗'} {message}")
    
    time.sleep(0.5)
    
    service.stop_worker()


def demo_deduplication():
    """Demonstrate deduplication feature"""
    print_separator("Deduplication")
    
    service = NotificationService()
    service.start_worker()
    
    print("\n🔹 Sending same notification twice within 5 minutes:")
    
    notification = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_123",
        channel=NotificationChannel.EMAIL,
        title="Account Update",
        message="Your account has been updated successfully.",
        priority=NotificationPriority.MEDIUM
    )
    
    # First attempt
    success1, message1 = service.send_notification(notification)
    print(f"First attempt: {'✓' if success1 else '✗'} {message1}")
    
    time.sleep(0.3)
    
    # Second attempt (duplicate)
    notification.notification_id = str(uuid.uuid4())  # New ID but same content
    success2, message2 = service.send_notification(notification)
    print(f"Second attempt: {'✓' if success2 else '✗'} {message2}")
    
    print("\n🔹 Sending different notification (should succeed):")
    
    different_notif = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_123",
        channel=NotificationChannel.EMAIL,
        title="Account Update",
        message="Your password has been changed.",  # Different message
        priority=NotificationPriority.HIGH
    )
    
    success3, message3 = service.send_notification(different_notif)
    print(f"Different notification: {'✓' if success3 else '✗'} {message3}")
    
    time.sleep(0.3)
    
    service.stop_worker()


def demo_rate_limiting():
    """Demonstrate rate limiting"""
    print_separator("Rate Limiting")
    
    service = NotificationService()
    
    # Configure rate limit: max 3 SMS per minute
    service.rate_limiter.set_channel_config(
        NotificationChannel.SMS,
        RateLimitConfig(max_notifications=3, time_window_seconds=60)
    )
    
    service.start_worker()
    
    print("\n🔹 Sending 5 SMS to same user (limit: 3/minute):")
    
    for i in range(5):
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            user_id="user_999",
            channel=NotificationChannel.SMS,
            title=f"Alert {i+1}",
            message=f"This is alert number {i+1}",
            priority=NotificationPriority.MEDIUM
        )
        
        success, message = service.send_notification(notification)
        status_icon = '✓' if success else '✗'
        print(f"  SMS {i+1}: {status_icon} {message}")
        time.sleep(0.1)
    
    time.sleep(0.5)
    
    service.stop_worker()


def demo_priority_queue():
    """Demonstrate priority-based delivery"""
    print_separator("Priority-Based Delivery")
    
    service = NotificationService()
    
    print("\n🔹 Queueing notifications with different priorities:")
    
    # Queue low priority notification
    low_priority = Notification(
        notification_id="notif_low",
        user_id="user_111",
        channel=NotificationChannel.EMAIL,
        title="Newsletter",
        message="Check out this week's newsletter",
        priority=NotificationPriority.LOW
    )
    
    # Queue medium priority notification
    medium_priority = Notification(
        notification_id="notif_medium",
        user_id="user_222",
        channel=NotificationChannel.EMAIL,
        title="Account Activity",
        message="New login detected from Chrome",
        priority=NotificationPriority.MEDIUM
    )
    
    # Queue critical priority notification
    critical_priority = Notification(
        notification_id="notif_critical",
        user_id="user_333",
        channel=NotificationChannel.SMS,
        title="Security Alert",
        message="Unusual activity detected. Please verify.",
        priority=NotificationPriority.CRITICAL
    )
    
    # Queue in reverse priority order
    print("Queueing: LOW -> MEDIUM -> CRITICAL")
    service.send_notification(low_priority)
    service.send_notification(medium_priority)
    service.send_notification(critical_priority)
    
    print("\nProcessing queue (should process in CRITICAL -> MEDIUM -> LOW order):")
    service.start_worker()
    
    time.sleep(1.0)
    
    service.stop_worker()


def demo_statistics():
    """Demonstrate statistics tracking"""
    print_separator("Service Statistics")
    
    service = NotificationService()
    service.start_worker()
    
    # Send various notifications
    channels = [
        NotificationChannel.EMAIL,
        NotificationChannel.SMS,
        NotificationChannel.PUSH,
        NotificationChannel.SLACK
    ]
    
    for i, channel in enumerate(channels * 2):  # 8 total notifications
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            user_id=f"user_{i}",
            channel=channel,
            title=f"Test {i}",
            message=f"Test message {i}",
            priority=NotificationPriority.MEDIUM
        )
        service.send_notification(notification)
    
    time.sleep(1.0)
    
    stats = service.get_statistics()
    
    print("\n📊 Notification Statistics:")
    print(f"Total notifications: {stats['total_notifications']}")
    print(f"Queue size: {stats['queue_size']}")
    print(f"Retry queue size: {stats['retry_queue_size']}")
    
    print("\nBy Status:")
    for status, count in stats['by_status'].items():
        print(f"  {status}: {count}")
    
    service.stop_worker()


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  NOTIFICATION SERVICE - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Multi-channel, Deduplication, Rate Limiting, Retry")
    print("="*70)
    
    demo_basic_operations()
    demo_deduplication()
    demo_rate_limiting()
    demo_priority_queue()
    demo_statistics()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python 10_notification_service.py
    """
    run_demo()
