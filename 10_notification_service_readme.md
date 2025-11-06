# Notification Service - Interview Preparation Guide

**Target Audience**: Software Engineers with 2-5 years of experience  
**Focus**: Multi-channel delivery, deduplication, rate limiting, retry mechanisms  
**Estimated Study Time**: 3-4 hours

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
3. [Critical Knowledge Points](#critical-knowledge-points)
4. [Expected Interview Questions & Answers](#expected-interview-questions--answers)
5. [Testing Strategy](#testing-strategy)
6. [Production Considerations](#production-considerations)

---

## Problem Statement

Design a notification service that can:
- Send notifications through multiple channels (Email, SMS, Push, Slack)
- Queue notifications with priority-based delivery
- Deduplicate notifications to prevent spam
- Rate limit notifications per user per channel
- Retry failed deliveries with exponential backoff
- Track delivery status and generate statistics

**Core Challenge**: How do you design a reliable, scalable notification system that prevents spam while ensuring critical notifications are delivered promptly?

---

## Step-by-Step Implementation Guide

### Phase 1: Channel Abstraction with Strategy Pattern (15-20 minutes)

**What to do**:
```python
class NotificationChannelProvider(ABC):
    @abstractmethod
    def send(self, notification: Notification) -> Tuple[bool, str]:
        pass
    
    @abstractmethod
    def validate_recipient(self, user_id: str) -> bool:
        pass

class EmailProvider(NotificationChannelProvider):
    def send(self, notification: Notification):
        # SMTP integration
        return True, "Email sent"

class SMSProvider(NotificationChannelProvider):
    def send(self, notification: Notification):
        # SMS gateway (Twilio, AWS SNS)
        return True, "SMS sent"
```

**Why Strategy Pattern**:
- Each channel has different delivery mechanism (SMTP vs HTTP API)
- Easy to add new channels without modifying existing code
- Testable - can mock individual providers
- Different error handling per channel

**Common mistake**: Putting all channel logic in one class with if/else statements.

---

### Phase 2: Deduplication Algorithm (15-20 minutes)

**What to do**:
```python
class NotificationDeduplicator:
    def __init__(self, window_seconds: int = 300):
        self.sent_notifications: Dict[str, float] = {}  # hash -> timestamp
        self.window_seconds = window_seconds
    
    def is_duplicate(self, notification: Notification) -> bool:
        dedup_key = self._generate_hash(notification)
        current_time = time.time()
        
        if dedup_key in self.sent_notifications:
            last_sent = self.sent_notifications[dedup_key]
            if current_time - last_sent < self.window_seconds:
                return True  # Duplicate within window
        
        self.sent_notifications[dedup_key] = current_time
        return False
    
    def _generate_hash(self, notification):
        content = f"{notification.user_id}:{notification.channel}:{notification.message}"
        return hashlib.sha256(content.encode()).hexdigest()
```

**Why Time-Window Deduplication**:
- Prevents same notification sent multiple times in short period
- Configurable window (e.g., 5 minutes for transactional, 1 hour for promotional)
- Hash-based detection is O(1) lookup
- Automatic cleanup of old entries

**Critical Detail**: Hash includes user_id, channel, AND content - same message to different users is not duplicate.

**When it fails**: If content has timestamps or dynamic IDs embedded, hash changes each time.

---

### Phase 3: Rate Limiting with Sliding Window (20-25 minutes)

**What to do**:
```python
class NotificationRateLimiter:
    def __init__(self, max_notifications: int, window_seconds: int):
        self.max_notifications = max_notifications
        self.window_seconds = window_seconds
        # user:channel -> [timestamps]
        self.notification_log: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, notification: Notification) -> bool:
        key = f"{notification.user_id}:{notification.channel}"
        current_time = time.time()
        log = self.notification_log[key]
        
        # Remove old timestamps outside window
        cutoff = current_time - self.window_seconds
        while log and log[0] < cutoff:
            log.popleft()
        
        # Check limit
        if len(log) >= self.max_notifications:
            return False  # Rate limit exceeded
        
        log.append(current_time)
        return True
```

**Why Sliding Window**:
- **Accuracy**: Counts exact notifications in rolling time window
- **Fairness**: No boundary issues like fixed window
- **Trade-off**: More memory (stores all timestamps)

**Alternative: Token Bucket**:
- Allows bursts
- Less memory (just token count)
- But less accurate for strict limits

**Interview Tip**: Mention that in production, you'd use Redis with sorted sets for distributed rate limiting.

---

### Phase 4: Priority Queue and Retry Mechanism (25-30 minutes)

**What to do**:
```python
from queue import PriorityQueue

class NotificationPriority(Enum):
    CRITICAL = 0  # Lowest number = highest priority
    HIGH = 1
    MEDIUM = 2
    LOW = 3

# Notification class implements __lt__ for comparison
@dataclass
class Notification:
    priority: NotificationPriority
    # ... other fields
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

# Service uses PriorityQueue
class NotificationService:
    def __init__(self):
        self.delivery_queue = PriorityQueue()
    
    def send_notification(self, notification):
        self.delivery_queue.put(notification)  # Auto-sorted by priority
```

**Retry with Exponential Backoff**:
```python
@dataclass
class RetryableNotification:
    notification: Notification
    attempt_count: int = 0
    max_attempts: int = 3
    
    def calculate_next_retry(self) -> float:
        base_delay = 60  # seconds
        delay = base_delay * (2 ** self.attempt_count)
        max_delay = 3600  # cap at 1 hour
        return time.time() + min(delay, max_delay)
```

**Why Exponential Backoff**:
- Prevents thundering herd (all retries hitting at once)
- Gives external services time to recover
- Standard pattern: delay = base * 2^attempt

**Interview Tip**: Explain that priority queue ensures critical notifications (2FA codes) are sent before newsletters.

---

### Phase 5: Background Worker and Status Tracking (15-20 minutes)

**What to do**:
```python
class NotificationService:
    def __init__(self):
        self.delivery_queue = PriorityQueue()
        self.delivery_history: Dict[str, NotificationResult] = {}
        self.running = False
    
    def start_worker(self):
        self.running = True
        Thread(target=self._worker_loop, daemon=True).start()
    
    def _worker_loop(self):
        while self.running:
            if not self.delivery_queue.empty():
                notification = self.delivery_queue.get()
                self._process_delivery(notification)
            
            # Also process retry queue
            for retry_notif in self.retry_queue.get_ready():
                self._process_delivery(retry_notif.notification)
            
            time.sleep(0.1)  # Prevent busy waiting
    
    def get_delivery_status(self, notification_id: str):
        return self.delivery_history.get(notification_id)
```

**Error Recovery Strategy**:
- Track all delivery attempts with timestamps
- Store failure reasons for debugging
- Support manual retry for failed notifications
- Alert on persistent failures

---

## Critical Knowledge Points

### 1. Why Strategy Pattern for Channels?

**Without Pattern**:
```python
def send_notification(channel, notification):
    if channel == "email":
        # Email logic
    elif channel == "sms":
        # SMS logic
    elif channel == "push":
        # Push logic
    # Adding new channel requires modifying this function
```

**With Pattern**:
```python
class EmailProvider(NotificationChannelProvider):
    def send(self, notification):
        # Email-specific logic
        pass

# Just add new provider, no modification needed
class WhatsAppProvider(NotificationChannelProvider):
    def send(self, notification):
        # WhatsApp logic
        pass
```

**Benefits**:
- **Open/Closed Principle**: Open for extension, closed for modification
- **Single Responsibility**: Each provider handles one channel
- **Testability**: Mock providers easily for testing
- **Configuration**: Load providers dynamically from config

---

### 2. Deduplication Algorithm Explained

**Algorithm**:
```python
def is_duplicate(notification):
    # 1. Generate content hash
    hash_key = sha256(f"{user_id}:{channel}:{content}")
    
    # 2. Check if sent recently
    if hash_key in sent_cache:
        last_sent_time = sent_cache[hash_key]
        if now - last_sent_time < dedup_window:
            return True  # Duplicate!
    
    # 3. Record this notification
    sent_cache[hash_key] = now
    return False
```

**Time**: O(1) average for hash lookup  
**Space**: O(n * w) where n=users, w=window size

**Why it works**: Content hash uniquely identifies notification. Same hash within time window = duplicate.

**When it fails**: 
- Dynamic content (timestamps in message)
- Different users should get same notification (use without user_id in hash)

**Alternative: Bloom Filters**:
```python
# More memory efficient but probabilistic
bloom_filter = BloomFilter(size=10000, hash_count=3)

def is_duplicate_bloom(notification):
    key = generate_key(notification)
    if bloom_filter.might_contain(key):
        return True  # Might be duplicate (false positives possible)
    bloom_filter.add(key)
    return False
```

**Trade-off**: Bloom filters save memory but may have false positives (incorrectly mark as duplicate).

---

### 3. Rate Limiting: Sliding Window vs Token Bucket

**Sliding Window (Our Implementation)**:
```python
# Stores all timestamps
log = [t1, t2, t3, t4, t5]  # Last 5 notifications

# On new notification:
1. Remove timestamps older than window
2. Check if count < limit
3. Add current timestamp
```

**Pros**: Accurate, fair, no boundary issues  
**Cons**: More memory (stores all timestamps)

**Token Bucket Alternative**:
```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
    
    def is_allowed(self):
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
```

**Pros**: Memory efficient, allows bursts  
**Cons**: Less accurate at boundaries

**Interview Insight**: Use sliding window for strict limits (security alerts), token bucket for flexible limits (newsletters).

---

### 4. Exponential Backoff for Retries

**Problem**: Service is down, 1000 notifications fail simultaneously. All retry at same time → thundering herd problem.

**Solution**: Exponential backoff with jitter
```python
def calculate_retry_delay(attempt):
    base_delay = 60  # 1 minute
    max_delay = 3600  # 1 hour
    
    # Exponential: 60s, 120s, 240s, 480s, ...
    delay = base_delay * (2 ** attempt)
    delay = min(delay, max_delay)
    
    # Add jitter to spread out retries
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter
```

**Retry Schedule Example**:
- Attempt 1: Immediate
- Attempt 2: 1 minute later
- Attempt 3: 2 minutes later
- Attempt 4: 4 minutes later
- Attempt 5: 8 minutes later
- Give up after 5 attempts

**Why it works**: Spreads retry load over time, gives downstream services time to recover.

---

## Expected Interview Questions & Answers

### Q1: How would you scale this notification service to handle millions of notifications per day?

**Answer**:
To scale to millions of notifications, I'd implement a distributed architecture with these components:

**1. Load Balancing**: Multiple notification service instances behind a load balancer
```python
# Each instance handles subset of users via consistent hashing
instance = hash(user_id) % num_instances
```

**2. Message Queue**: Replace in-memory PriorityQueue with distributed queue (RabbitMQ, Kafka, AWS SQS)
```python
# Producer (API servers)
producer.send_message(
    queue="notifications",
    message=notification.to_json(),
    priority=notification.priority.value
)

# Consumer (notification workers)
for message in consumer.consume(queue="notifications"):
    notification = Notification.from_json(message)
    process_notification(notification)
```

**3. Distributed Deduplication**: Use Redis with TTL
```python
def is_duplicate_distributed(notification):
    dedup_key = notification.get_hash()
    redis_key = f"notif:dedup:{dedup_key}"
    
    # SETNX: Set if not exists (atomic)
    if not redis.setnx(redis_key, "1"):
        return True  # Already exists = duplicate
    
    # Set expiration (dedup window)
    redis.expire(redis_key, dedup_window_seconds)
    return False
```

**4. Distributed Rate Limiting**: Redis sorted sets
```python
def is_allowed_distributed(user_id, channel):
    key = f"ratelimit:{user_id}:{channel}"
    now = time.time()
    window_start = now - window_seconds
    
    # Remove old timestamps
    redis.zremrangebyscore(key, 0, window_start)
    
    # Check count
    count = redis.zcard(key)
    if count >= max_notifications:
        return False
    
    # Add current timestamp
    redis.zadd(key, {str(uuid.uuid4()): now})
    redis.expire(key, window_seconds)
    return True
```

**Key Points**:
1. Horizontal scaling with multiple workers
2. Persistent queue survives restarts
3. Redis for shared state (dedup, rate limiting)
4. Database for delivery history and analytics

**Follow-up**: How do you handle Redis failure? Use local cache as fallback, better to send duplicate than miss critical notification.

---

### Q2: How would you ensure exactly-once delivery for critical notifications?

**Answer**:
Exactly-once delivery is challenging in distributed systems. I'd implement an idempotency layer:

**1. Idempotency Key**:
```python
@dataclass
class Notification:
    idempotency_key: str  # Client-generated unique ID
    # ... other fields

def send_notification(notification):
    # Check if already processed
    if redis.exists(f"idempotent:{notification.idempotency_key}"):
        return get_cached_result(notification.idempotency_key)
    
    # Process and cache result
    result = actually_send_notification(notification)
    redis.setex(
        f"idempotent:{notification.idempotency_key}",
        3600,  # 1 hour TTL
        result.to_json()
    )
    return result
```

**2. Database Transaction with Status**:
```python
def send_with_transaction(notification):
    with database.transaction():
        # Record intent
        db.insert("notifications", {
            "id": notification.id,
            "status": "PROCESSING",
            "created_at": now()
        })
        
        # Send notification
        result = provider.send(notification)
        
        # Update status
        db.update("notifications", 
            notification.id,
            {"status": "SENT", "sent_at": now()}
        )
```

**3. Delivery Confirmation**:
```python
# For critical notifications, require acknowledgment
def send_with_confirmation(notification):
    result = provider.send(notification)
    
    if result.success:
        # Wait for delivery receipt (webhook from provider)
        wait_for_delivery_receipt(
            notification.id,
            timeout=300  # 5 minutes
        )
```

**Trade-offs**:
- True exactly-once is impossible in distributed systems
- We can achieve at-most-once (with dedup) or at-least-once (with retry)
- For critical notifications, prefer at-least-once with idempotency

---

### Q3: How do you handle notifications for users in different time zones?

**Answer**:
Implement scheduled delivery with timezone awareness:

**Implementation**:
```python
@dataclass
class Notification:
    # ... other fields
    scheduled_time: Optional[datetime] = None  # User's local time
    user_timezone: str = "UTC"

class NotificationScheduler:
    def schedule_notification(self, notification: Notification):
        if notification.scheduled_time:
            # Convert user's local time to UTC
            user_tz = pytz.timezone(notification.user_timezone)
            local_time = user_tz.localize(notification.scheduled_time)
            utc_time = local_time.astimezone(pytz.UTC)
            
            # Schedule for delivery at UTC time
            self.scheduled_queue.add(notification, deliver_at=utc_time)
        else:
            # Send immediately
            self.send_now(notification)
    
    def process_scheduled_notifications(self):
        """Background job runs every minute"""
        now = datetime.now(pytz.UTC)
        
        # Get notifications due for delivery
        due_notifications = self.scheduled_queue.get_due(now)
        
        for notification in due_notifications:
            self.delivery_queue.put(notification)
```

**Use Cases**:
1. **Marketing emails**: Send at 9 AM user's local time
2. **Daily summaries**: Deliver at preferred time
3. **Reminders**: Respect user's timezone

**Database Schema**:
```sql
CREATE TABLE notifications (
    id UUID PRIMARY KEY,
    user_id UUID,
    scheduled_utc TIMESTAMP,  -- Stored in UTC
    user_timezone VARCHAR(50),
    status VARCHAR(20)
);

-- Index for efficient querying
CREATE INDEX idx_scheduled_pending 
ON notifications(scheduled_utc, status)
WHERE status = 'SCHEDULED';
```

---

### Q4: How would you implement notification preferences (opt-in/opt-out)?

**Answer**:
Create a flexible preference system with notification types and channels:

**Implementation**:
```python
class NotificationType(Enum):
    SECURITY_ALERT = "security_alert"  # Always send
    ACCOUNT_UPDATE = "account_update"
    MARKETING = "marketing"
    SOCIAL = "social"  # Comments, likes, etc.

@dataclass
class UserPreferences:
    user_id: str
    preferences: Dict[NotificationType, Set[NotificationChannel]]
    
    def is_allowed(self, notification_type, channel):
        if notification_type == NotificationType.SECURITY_ALERT:
            return True  # Security alerts always allowed
        
        allowed_channels = self.preferences.get(notification_type, set())
        return channel in allowed_channels

class NotificationService:
    def __init__(self):
        self.preference_store = PreferenceStore()
    
    def send_notification(self, notification: Notification):
        # Check user preferences
        prefs = self.preference_store.get(notification.user_id)
        
        if not prefs.is_allowed(notification.type, notification.channel):
            return False, "User opted out of this notification type"
        
        # Continue with delivery
        return self._deliver(notification)
```

**Database Schema**:
```sql
CREATE TABLE user_notification_preferences (
    user_id UUID,
    notification_type VARCHAR(50),
    channel VARCHAR(20),
    enabled BOOLEAN DEFAULT true,
    PRIMARY KEY (user_id, notification_type, channel)
);

-- Example data:
-- user_123, MARKETING, EMAIL, false  (opted out)
-- user_123, MARKETING, SMS, false    (opted out)
-- user_123, ACCOUNT_UPDATE, EMAIL, true
```

**Key Points**:
1. Security/critical notifications ignore preferences
2. Granular control: type + channel combination
3. Default opt-in for essential, opt-out for marketing
4. Legal compliance (GDPR, CAN-SPAM)

---

### Q5: How do you prevent notification fatigue?

**Answer**:
Implement intelligent notification aggregation and digest:

**1. Batching Similar Notifications**:
```python
class NotificationBatcher:
    def __init__(self, batch_window: int = 300):
        self.batch_window = batch_window  # 5 minutes
        self.pending_batches: Dict[str, List[Notification]] = {}
    
    def add_notification(self, notification: Notification):
        batch_key = f"{notification.user_id}:{notification.type}"
        
        if batch_key not in self.pending_batches:
            # Start new batch, schedule send
            self.pending_batches[batch_key] = []
            schedule_batch_send(batch_key, delay=self.batch_window)
        
        self.pending_batches[batch_key].append(notification)
    
    def send_batch(self, batch_key: str):
        notifications = self.pending_batches.pop(batch_key)
        
        if len(notifications) == 1:
            # Send individually
            send_notification(notifications[0])
        else:
            # Send as digest
            digest = create_digest(notifications)
            send_notification(digest)

def create_digest(notifications):
    """Combine multiple notifications into one"""
    return Notification(
        title=f"You have {len(notifications)} new updates",
        message=summarize_notifications(notifications),
        # ... other fields
    )
```

**2. Quiet Hours**:
```python
class QuietHoursManager:
    def should_send_now(self, notification, user_prefs):
        if notification.priority == NotificationPriority.CRITICAL:
            return True  # Always send critical
        
        user_time = get_user_local_time(notification.user_id)
        hour = user_time.hour
        
        # Don't send non-critical between 10 PM - 8 AM
        if hour >= 22 or hour < 8:
            return False
        
        return True
```

**3. Frequency Capping**:
```python
# Limit total notifications per day regardless of type
def check_daily_cap(user_id):
    key = f"daily_cap:{user_id}:{date.today()}"
    count = redis.incr(key)
    redis.expire(key, 86400)  # 24 hours
    
    MAX_DAILY_NOTIFICATIONS = 50
    return count <= MAX_DAILY_NOTIFICATIONS
```

**4. Priority-Based Suppression**:
```python
# Suppress low priority if user has many unread
def should_suppress(notification):
    if notification.priority in [CRITICAL, HIGH]:
        return False
    
    unread_count = get_unread_count(notification.user_id)
    if unread_count > 20:
        return True  # Too many unread, skip low priority
    
    return False
```

---

### Q6: How would you implement notification templates?

**Answer**:
Use template system with variable substitution and localization:

**Implementation**:
```python
@dataclass
class NotificationTemplate:
    template_id: str
    template_type: NotificationType
    title_template: str
    body_template: str
    supported_channels: Set[NotificationChannel]
    variables: List[str]  # Required variables

class TemplateEngine:
    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.load_templates()
    
    def render(self, template_id: str, variables: Dict[str, Any], 
               locale: str = "en") -> Tuple[str, str]:
        """Render template with variables"""
        template = self.templates[template_id]
        
        # Validate variables
        missing = set(template.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        # Render with locale
        title = self._render_string(template.title_template, variables, locale)
        body = self._render_string(template.body_template, variables, locale)
        
        return title, body
    
    def _render_string(self, template_str, variables, locale):
        # Simple variable substitution
        result = template_str
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

# Usage
class NotificationService:
    def send_from_template(self, user_id: str, template_id: str, 
                           variables: Dict[str, Any], channel: NotificationChannel):
        # Render template
        title, body = self.template_engine.render(
            template_id, 
            variables, 
            locale=get_user_locale(user_id)
        )
        
        # Create notification
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            user_id=user_id,
            channel=channel,
            title=title,
            message=body
        )
        
        return self.send_notification(notification)

# Example usage
service.send_from_template(
    user_id="user_123",
    template_id="welcome_email",
    variables={
        "first_name": "Alice",
        "signup_date": "Nov 6, 2025"
    },
    channel=NotificationChannel.EMAIL
)
```

**Template Storage**:
```python
# templates.json
{
    "welcome_email": {
        "type": "ACCOUNT_UPDATE",
        "channels": ["EMAIL"],
        "title": "Welcome to {product_name}, {first_name}!",
        "body": "Thanks for joining on {signup_date}. Get started here: {onboarding_link}",
        "variables": ["product_name", "first_name", "signup_date", "onboarding_link"]
    },
    "password_reset": {
        "type": "SECURITY_ALERT",
        "channels": ["EMAIL", "SMS"],
        "title": "Password Reset Request",
        "body": "Reset code: {reset_code}. Expires in {expiry_minutes} minutes.",
        "variables": ["reset_code", "expiry_minutes"]
    }
}
```

**Benefits**:
1. Centralized content management
2. A/B testing different templates
3. Easy localization
4. Non-engineers can update content

---

### Q7: How do you monitor notification service health?

**Answer**:
Implement comprehensive monitoring and alerting:

**Metrics to Track**:
```python
class NotificationMetrics:
    def __init__(self):
        self.metrics = {
            "total_sent": 0,
            "total_failed": 0,
            "total_queued": 0,
            "by_channel": defaultdict(lambda: {"sent": 0, "failed": 0}),
            "by_priority": defaultdict(int),
            "average_delivery_time": 0,
            "retry_count": 0
        }
    
    def record_sent(self, notification, delivery_time):
        self.metrics["total_sent"] += 1
        self.metrics["by_channel"][notification.channel.value]["sent"] += 1
        self.update_average_delivery_time(delivery_time)
    
    def record_failed(self, notification, error):
        self.metrics["total_failed"] += 1
        self.metrics["by_channel"][notification.channel.value]["failed"] += 1
        log_failure(notification, error)
    
    def get_health_score(self) -> float:
        """Calculate service health (0-100)"""
        total = self.metrics["total_sent"] + self.metrics["total_failed"]
        if total == 0:
            return 100.0
        
        success_rate = (self.metrics["total_sent"] / total) * 100
        return success_rate
```

**Alerting Rules**:
```python
class NotificationAlerting:
    def check_alerts(self, metrics):
        alerts = []
        
        # Alert if success rate drops below 95%
        success_rate = metrics.get_health_score()
        if success_rate < 95:
            alerts.append({
                "severity": "HIGH",
                "message": f"Notification success rate: {success_rate:.2f}%"
            })
        
        # Alert if queue size too large
        queue_size = metrics.metrics["total_queued"]
        if queue_size > 10000:
            alerts.append({
                "severity": "MEDIUM",
                "message": f"Queue size: {queue_size} (backlog detected)"
            })
        
        # Alert if specific channel failing
        for channel, stats in metrics.metrics["by_channel"].items():
            total = stats["sent"] + stats["failed"]
            if total > 0:
                channel_success_rate = (stats["sent"] / total) * 100
                if channel_success_rate < 80:
                    alerts.append({
                        "severity": "HIGH",
                        "message": f"{channel} channel failing: {channel_success_rate:.2f}%"
                    })
        
        return alerts
```

**Dashboards**:
1. **Real-time metrics**: Sent/failed per minute
2. **Channel health**: Success rate per channel
3. **Queue depth**: Monitor for backlogs
4. **Latency**: P50, P95, P99 delivery times
5. **Error rates**: By error type and channel

---

## Testing Strategy

### Unit Tests

**Test channel providers independently**:
```python
def test_email_provider_success():
    provider = EmailProvider()
    notification = Notification(
        notification_id="test_1",
        user_id="user_123",
        channel=NotificationChannel.EMAIL,
        title="Test",
        message="Test message"
    )
    
    success, message = provider.send(notification)
    assert success == True
    assert "sent" in message.lower()

def test_email_provider_invalid_recipient():
    provider = EmailProvider()
    assert provider.validate_recipient("invalid_user") == False
```

**Test deduplication**:
```python
def test_deduplication_within_window():
    deduplicator = NotificationDeduplicator(window_seconds=60)
    
    notification = create_test_notification()
    
    # First attempt - not duplicate
    assert deduplicator.is_duplicate(notification) == False
    
    # Second attempt within window - duplicate
    assert deduplicator.is_duplicate(notification) == True
    
def test_deduplication_outside_window():
    deduplicator = NotificationDeduplicator(window_seconds=1)
    notification = create_test_notification()
    
    assert deduplicator.is_duplicate(notification) == False
    time.sleep(2)  # Wait for window to expire
    assert deduplicator.is_duplicate(notification) == False
```

**Test rate limiting**:
```python
def test_rate_limiting():
    config = RateLimitConfig(max_notifications=3, time_window_seconds=60)
    limiter = NotificationRateLimiter(config)
    
    notification = create_test_notification(user_id="user_123")
    
    # First 3 should succeed
    for i in range(3):
        allowed, msg = limiter.is_allowed(notification)
        assert allowed == True
    
    # 4th should fail
    allowed, msg = limiter.is_allowed(notification)
    assert allowed == False
    assert "rate limit" in msg.lower()
```

---

### Integration Tests

**Test complete flow**:
```python
def test_end_to_end_delivery():
    service = NotificationService()
    service.start_worker()
    
    notification = Notification(
        notification_id=str(uuid.uuid4()),
        user_id="user_123",
        channel=NotificationChannel.EMAIL,
        title="Test",
        message="Integration test",
        priority=NotificationPriority.HIGH
    )
    
    success, msg = service.send_notification(notification)
    assert success == True
    
    # Wait for processing
    time.sleep(1)
    
    # Check status
    status = service.get_delivery_status(notification.notification_id)
    assert status.status == NotificationStatus.SENT
    
    service.stop_worker()

def test_retry_on_failure():
    service = NotificationService()
    
    # Mock provider to fail first attempt
    original_send = service.providers[NotificationChannel.EMAIL].send
    attempt_count = [0]
    
    def mock_send(notification):
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            return False, "Temporary failure"
        return original_send(notification)
    
    service.providers[NotificationChannel.EMAIL].send = mock_send
    service.start_worker()
    
    notification = create_test_notification()
    service.send_notification(notification)
    
    time.sleep(2)  # Wait for retry
    
    # Should have retried and succeeded
    assert attempt_count[0] >= 2
    
    service.stop_worker()
```

---

### Load Testing

**Simulate high volume**:
```python
import threading
import time

def test_concurrent_notifications():
    service = NotificationService()
    service.start_worker()
    
    def send_batch(user_id, count):
        for i in range(count):
            notification = Notification(
                notification_id=str(uuid.uuid4()),
                user_id=user_id,
                channel=random.choice(list(NotificationChannel)),
                title=f"Test {i}",
                message=f"Message {i}",
                priority=random.choice(list(NotificationPriority))
            )
            service.send_notification(notification)
    
    # Simulate 100 users sending 10 notifications each
    threads = []
    for user_id in range(100):
        thread = threading.Thread(target=send_batch, args=(f"user_{user_id}", 10))
        threads.append(thread)
        thread.start()
    
    start_time = time.time()
    
    for thread in threads:
        thread.join()
    
    elapsed = time.time() - start_time
    
    # Wait for queue to process
    time.sleep(5)
    
    stats = service.get_statistics()
    print(f"Processed {stats['total_notifications']} notifications in {elapsed:.2f}s")
    print(f"Throughput: {stats['total_notifications']/elapsed:.2f} notifications/sec")
    
    service.stop_worker()
```

---

## Production Considerations

### 1. Persistence

**Current implementation**: In-memory only  
**Production needs**: Database for reliability

```python
class PersistentNotificationService:
    def __init__(self, db: Database, queue: MessageQueue):
        self.db = db
        self.queue = queue  # RabbitMQ, SQS, etc.
    
    def send_notification(self, notification: Notification):
        # 1. Persist to database
        self.db.insert("notifications", {
            "id": notification.notification_id,
            "user_id": notification.user_id,
            "channel": notification.channel.value,
            "status": "QUEUED",
            "created_at": notification.created_at,
            "priority": notification.priority.value,
            "payload": notification.to_json()
        })
        
        # 2. Enqueue for delivery
        self.queue.publish(
            exchange="notifications",
            routing_key=f"priority.{notification.priority.value}",
            message=notification.to_json()
        )
        
        return True, "Notification persisted and queued"
    
    def mark_as_sent(self, notification_id: str):
        self.db.update("notifications", notification_id, {
            "status": "SENT",
            "sent_at": datetime.now()
        })
```

**Schema**:
```sql
CREATE TABLE notifications (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    channel VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    priority INT NOT NULL,
    title TEXT,
    message TEXT,
    created_at TIMESTAMP NOT NULL,
    sent_at TIMESTAMP,
    failed_at TIMESTAMP,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    INDEX idx_user_status (user_id, status),
    INDEX idx_status_priority (status, priority)
);
```

---

### 2. Monitoring & Alerts

**Implement comprehensive monitoring**:
```python
class NotificationMonitor:
    def __init__(self, metrics_client):
        self.metrics = metrics_client  # Prometheus, CloudWatch, etc.
    
    def track_notification_sent(self, notification, duration_ms):
        self.metrics.increment(
            "notifications.sent",
            tags={
                "channel": notification.channel.value,
                "priority": notification.priority.value
            }
        )
        self.metrics.histogram(
            "notifications.delivery_time_ms",
            duration_ms,
            tags={"channel": notification.channel.value}
        )
    
    def track_notification_failed(self, notification, error):
        self.metrics.increment(
            "notifications.failed",
            tags={
                "channel": notification.channel.value,
                "error_type": type(error).__name__
            }
        )
    
    def track_queue_depth(self, depth):
        self.metrics.gauge("notifications.queue_depth", depth)
```

**Alert Configuration**:
```yaml
alerts:
  - name: HighFailureRate
    condition: (failed / total) > 0.05
    severity: HIGH
    message: "Notification failure rate above 5%"
  
  - name: QueueBacklog
    condition: queue_depth > 10000
    severity: MEDIUM
    message: "Notification queue has {queue_depth} pending"
  
  - name: ChannelDown
    condition: channel_success_rate < 0.5
    severity: CRITICAL
    message: "{channel} channel success rate below 50%"
```

---

### 3. Security

**Key concerns**:
1. **PII Protection**: Encrypt notification content
2. **Authentication**: Verify sender identity
3. **Rate Limiting**: Prevent abuse
4. **Audit Trail**: Log all notifications

```python
class SecureNotificationService(NotificationService):
    def __init__(self, encryption_key: bytes):
        super().__init__()
        self.cipher = Fernet(encryption_key)
    
    def send_notification(self, notification: Notification, api_key: str):
        # 1. Authenticate request
        if not self.verify_api_key(api_key):
            return False, "Unauthorized"
        
        # 2. Validate input
        if not self.validate_notification(notification):
            return False, "Invalid notification"
        
        # 3. Encrypt sensitive data
        notification.message = self.encrypt_message(notification.message)
        
        # 4. Audit log
        self.audit_log.record({
            "action": "notification_sent",
            "user_id": notification.user_id,
            "channel": notification.channel.value,
            "timestamp": time.time(),
            "api_key": hash(api_key)
        })
        
        # 5. Send
        return super().send_notification(notification)
    
    def encrypt_message(self, message: str) -> str:
        return self.cipher.encrypt(message.encode()).decode()
    
    def decrypt_message(self, encrypted: str) -> str:
        return self.cipher.decrypt(encrypted.encode()).decode()
```

---

### 4. Scalability

**Distributed architecture**:
```python
class DistributedNotificationService:
    def __init__(self, redis_client, message_queue):
        self.redis = redis_client
        self.queue = message_queue
        self.local_cache = {}
    
    def send_notification(self, notification):
        # 1. Check deduplication (distributed)
        if self.is_duplicate_distributed(notification):
            return False, "Duplicate"
        
        # 2. Check rate limit (distributed)
        if not self.check_rate_limit_distributed(notification):
            return False, "Rate limited"
        
        # 3. Enqueue to distributed queue
        self.queue.enqueue(notification, priority=notification.priority.value)
        
        return True, "Queued"
    
    def is_duplicate_distributed(self, notification):
        key = f"dedup:{notification.get_dedup_key()}"
        
        # Try Redis first
        try:
            if not self.redis.setnx(key, "1"):
                return True
            self.redis.expire(key, 300)  # 5 min window
            return False
        except RedisError:
            # Fallback to local cache
            return self.is_duplicate_local(notification)
```

**Horizontal Scaling**:
- Multiple worker instances consume from shared queue
- Redis for shared state (dedup, rate limiting)
- Load balancer distributes API requests
- Each worker handles subset of channels

---

## Summary

### Do's ✅
- Use Strategy Pattern for different notification channels
- Implement deduplication to prevent spam
- Apply rate limiting per user per channel
- Use priority queue for important notifications
- Implement retry with exponential backoff
- Track delivery status for debugging

### Don'ts ❌
- Don't send notifications synchronously (use queue)
- Don't ignore user preferences and opt-outs
- Don't retry indefinitely (set max attempts)
- Don't store sensitive data in logs
- Don't send during quiet hours (respect timezones)

### Key Takeaways
1. **Strategy Pattern**: Essential for supporting multiple channels with different delivery mechanisms
2. **Deduplication**: Time-window based with content hashing prevents duplicate notifications
3. **Rate Limiting**: Sliding window provides accurate limiting, token bucket allows bursts
4. **Retry Mechanism**: Exponential backoff prevents thundering herd and gives services time to recover
5. **Priority Queue**: Ensures critical notifications (2FA, security alerts) are delivered first

---

**Time to Master**: 3-4 hours  
**Difficulty**: Medium  
**Key Patterns**: Strategy, Observer, Queue  
**Critical Skills**: Distributed systems, rate limiting, deduplication algorithms, retry strategies
