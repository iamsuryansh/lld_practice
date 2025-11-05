'''
Design a Rate Limiter System (Low Level Design)

FUNCTIONAL REQUIREMENTS:
1. Rate limit API requests based on user_id
2. Support configurable rate limits (e.g., 100 requests per minute)
3. Support multiple rate limiting algorithms:
   - Token Bucket
   - Sliding Window Log
   - Fixed Window Counter
4. Return clear response when rate limit is exceeded
5. Allow requests when within rate limit

NON-FUNCTIONAL REQUIREMENTS:
1. Extensibility: Easy to add new rate limiting strategies
2. Maintainability: Clean code following SOLID principles
3. Performance: O(1) or O(log n) time complexity for allow_request()
4. Memory Efficient: Reasonable space complexity
5. Thread-safe (bonus if time permits)

DESIGN PATTERNS TO USE:
- Strategy Pattern (for different rate limiting algorithms)
- Factory Pattern (optional, for creating rate limiters)

INTERVIEW FLOW:
Step 1: Clarify Requirements ✓
Step 2: Design Core Classes & Interfaces ✓
Step 3: Implement Token Bucket Algorithm (Next)
Step 4: Implement Sliding Window Log
Step 5: Implement Fixed Window Counter
Step 6: Add Factory Pattern (optional)
Step 7: Write Test Cases
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime
from enum import Enum
from collections import deque
import time


# ==================== ENUMS ====================
# file: enums.py (in production)

class RateLimitStrategy(Enum):
    """Enum for different rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


# ==================== DATA CLASSES ====================
# file: models.py (in production)

@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting
    
    Attributes:
        max_requests: Maximum number of requests allowed
        time_window_seconds: Time window in seconds for the limit
    """
    max_requests: int
    time_window_seconds: int
    
    def __post_init__(self):
        """Validate configuration - Fail Fast Principle"""
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.time_window_seconds <= 0:
            raise ValueError("time_window_seconds must be positive")
    

@dataclass
class RateLimitResponse:
    """
    Response from rate limiter
    
    Attributes:
        allowed: Whether the request is allowed
        remaining_requests: Number of requests remaining (None if not applicable)
        retry_after_seconds: Seconds to wait before retrying (None if allowed)
        message: Human-readable message explaining the decision
    """
    allowed: bool
    remaining_requests: Optional[int] = None
    retry_after_seconds: Optional[int] = None
    message: Optional[str] = None


# ==================== ABSTRACT BASE CLASS ====================
# file: rate_limiter_interface.py (in production)

class RateLimiter(ABC):
    """
    Abstract base class for rate limiting strategies
    Strategy Pattern: Defines contract for all rate limiting algorithms
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter with configuration
        
        Args:
            config: RateLimitConfig object with rate limit settings
        """
        self.config = config
    
    @abstractmethod
    def allow_request(self, user_id: str) -> RateLimitResponse:
        """
        Check if request should be allowed for given user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            RateLimitResponse with decision and metadata
            
        Time Complexity: Should be O(1) or O(log n)
        Space Complexity: O(n) where n is number of unique users
        """
        pass
    
    @abstractmethod
    def reset(self, user_id: str) -> None:
        """
        Reset rate limit for a user (for testing/admin purposes)
        
        Args:
            user_id: Unique identifier for the user
        """
        pass


# ==================== CONCRETE IMPLEMENTATIONS ====================
# file: token_bucket.py (in production)

@dataclass
class TokenBucket:
    """
    Data class to store token bucket state for a user
    
    Attributes:
        tokens: Current number of tokens available
        last_refill_time: Last time tokens were refilled (Unix timestamp)
    """
    tokens: float
    last_refill_time: float


class TokenBucketRateLimiter(RateLimiter):
    """
    Token Bucket Rate Limiting Algorithm
    
    How it works:
    - Each user has a bucket with a maximum capacity (max_requests)
    - Tokens are added at a constant rate (refill rate)
    - Each request consumes 1 token
    - If bucket is empty, request is denied
    
    Pros:
    - Allows burst traffic (if bucket is full)
    - Smooth rate limiting
    - Memory efficient O(n) for n users
    
    Cons:
    - Slightly more complex than fixed window
    
    Time Complexity: O(1) per request
    Space Complexity: O(n) where n is number of unique users
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize Token Bucket rate limiter
        
        Args:
            config: RateLimitConfig with max_requests and time_window_seconds
        """
        super().__init__(config)
        # Storage: user_id -> TokenBucket
        self.buckets: Dict[str, TokenBucket] = {}
        # Calculate refill rate: tokens per second
        self.refill_rate = self.config.max_requests / self.config.time_window_seconds
    
    def _refill_tokens(self, user_id: str, current_time: float) -> None:
        """
        Refill tokens based on elapsed time (Lazy Refilling)
        
        Args:
            user_id: User identifier
            current_time: Current timestamp
        """
        bucket = self.buckets[user_id]
        
        # Calculate elapsed time since last refill
        elapsed_time = current_time - bucket.last_refill_time
        
        # Calculate tokens to add based on refill rate
        tokens_to_add = elapsed_time * self.refill_rate
        
        # Refill tokens (cap at max capacity)
        bucket.tokens = min(
            self.config.max_requests,
            bucket.tokens + tokens_to_add
        )
        
        # Update last refill time
        bucket.last_refill_time = current_time
    
    def allow_request(self, user_id: str) -> RateLimitResponse:
        """
        Check if request should be allowed for given user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            RateLimitResponse with decision and metadata
            
        Time Complexity: O(1)
        """
        current_time = time.time()
        
        # First request from this user - initialize bucket
        if user_id not in self.buckets:
            self.buckets[user_id] = TokenBucket(
                tokens=self.config.max_requests,
                last_refill_time=current_time
            )
        
        # Refill tokens based on elapsed time
        self._refill_tokens(user_id, current_time)
        
        bucket = self.buckets[user_id]
        
        # Check if we have at least 1 token
        if bucket.tokens >= 1.0:
            # Consume 1 token
            bucket.tokens -= 1.0
            
            return RateLimitResponse(
                allowed=True,
                remaining_requests=int(bucket.tokens),
                retry_after_seconds=None,
                message=f"Request allowed. {int(bucket.tokens)} requests remaining."
            )
        else:
            # Not enough tokens - calculate retry after
            tokens_needed = 1.0 - bucket.tokens
            retry_after = int(tokens_needed / self.refill_rate) + 1
            
            return RateLimitResponse(
                allowed=False,
                remaining_requests=0,
                retry_after_seconds=retry_after,
                message=f"Rate limit exceeded. Retry after {retry_after} seconds."
            )
    
    def reset(self, user_id: str) -> None:
        """
        Reset rate limit for a user
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self.buckets:
            del self.buckets[user_id]


# ==================== SLIDING WINDOW LOG ====================
# file: sliding_window_log.py (in production)

from typing import Deque

class SlidingWindowLogRateLimiter(RateLimiter):
    """
    Sliding Window Log Rate Limiting Algorithm
    
    How it works:
    - Store timestamps of all requests in a deque (double-ended queue)
    - On each request, remove timestamps older than time_window from left
    - Count remaining timestamps
    - If count < max_requests, allow request and append to right
    
    Pros:
    - Very accurate - no approximations
    - Precise rate limiting
    - No edge cases with window boundaries
    - O(1) removal from front using deque
    
    Cons:
    - Higher memory usage (stores all timestamps)
    - O(n) time complexity for cleanup in worst case
    - Not ideal for very high traffic
    
    Time Complexity: O(n) where n is requests in window (worst case: max_requests)
    Space Complexity: O(u × r) where u=users, r=max_requests
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize Sliding Window Log rate limiter
        
        Args:
            config: RateLimitConfig with max_requests and time_window_seconds
        """
        super().__init__(config)
        # Storage: user_id -> deque[timestamp]
        # Using deque for O(1) popleft() and append()
        self.request_logs: Dict[str, Deque[float]] = {}
    
    def _remove_old_timestamps(self, user_id: str, current_time: float) -> None:
        """
        Remove timestamps older than the time window
        Using deque.popleft() for efficient removal from front
        
        Args:
            user_id: User identifier
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.config.time_window_seconds
        
        # Remove old timestamps from the left (oldest first)
        while (self.request_logs[user_id] and 
               self.request_logs[user_id][0] <= cutoff_time):
            self.request_logs[user_id].popleft()
    
    def allow_request(self, user_id: str) -> RateLimitResponse:
        """
        Check if request should be allowed for given user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            RateLimitResponse with decision and metadata
            
        Time Complexity: O(n) where n is expired requests (amortized O(1))
        """
        current_time = time.time()
        
        # First request from this user - initialize log with deque
        if user_id not in self.request_logs:
            self.request_logs[user_id] = deque()
        
        # Remove expired timestamps
        self._remove_old_timestamps(user_id, current_time)
        
        request_count = len(self.request_logs[user_id])
        
        # Check if we can allow more requests
        if request_count < self.config.max_requests:
            # Add current timestamp to the right (newest)
            self.request_logs[user_id].append(current_time)
            
            remaining = self.config.max_requests - request_count - 1
            
            return RateLimitResponse(
                allowed=True,
                remaining_requests=remaining,
                retry_after_seconds=None,
                message=f"Request allowed. {remaining} requests remaining."
            )
        else:
            # Rate limit exceeded - calculate when oldest request expires
            oldest_timestamp = self.request_logs[user_id][0]
            retry_after = int(oldest_timestamp + self.config.time_window_seconds - current_time) + 1
            
            return RateLimitResponse(
                allowed=False,
                remaining_requests=0,
                retry_after_seconds=retry_after,
                message=f"Rate limit exceeded. Retry after {retry_after} seconds."
            )
    
    def reset(self, user_id: str) -> None:
        """
        Reset rate limit for a user
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self.request_logs:
            del self.request_logs[user_id]


# ==================== FIXED WINDOW COUNTER ====================
# file: fixed_window_counter.py (in production)

@dataclass
class FixedWindow:
    """
    Data class to store fixed window state for a user
    
    Attributes:
        count: Number of requests in current window
        window_start: Start time of current window (Unix timestamp)
    """
    count: int
    window_start: float


class FixedWindowCounterRateLimiter(RateLimiter):
    """
    Fixed Window Counter Rate Limiting Algorithm
    
    How it works:
    - Divide time into fixed windows of time_window_seconds
    - Count requests per window
    - Reset counter when new window starts
    - Allow request if count < max_requests
    
    Pros:
    - Simplest algorithm - easy to implement
    - O(1) time complexity - just counter increment
    - Low memory usage - only 2 values per user
    - Very fast - minimal computation
    
    Cons:
    - Boundary problem - can allow 2× requests at window edges
    - Not accurate for strict rate limiting
    - Unfair - early requests get advantage
    
    Example of boundary problem:
    Max 5 req/min:
      Window 1: [____🟢🟢🟢🟢🟢] (5 at 59s)
      Window 2: [🟢🟢🟢🟢🟢____] (5 at 61s)
      → 10 requests in 2 seconds!
    
    Time Complexity: O(1) per request
    Space Complexity: O(n) where n is number of unique users
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize Fixed Window Counter rate limiter
        
        Args:
            config: RateLimitConfig with max_requests and time_window_seconds
        """
        super().__init__(config)
        # Storage: user_id -> FixedWindow
        self.windows: Dict[str, FixedWindow] = {}
    
    def _get_window_start(self, current_time: float) -> float:
        """
        Calculate the start time of the current window
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Start timestamp of current window
        """
        # Floor division to get window start
        # Example: time=125s, window=60s → 120s (start of window)
        return (current_time // self.config.time_window_seconds) * self.config.time_window_seconds
    
    def allow_request(self, user_id: str) -> RateLimitResponse:
        """
        Check if request should be allowed for given user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            RateLimitResponse with decision and metadata
            
        Time Complexity: O(1)
        """
        current_time = time.time()
        current_window_start = self._get_window_start(current_time)
        
        # First request from this user - initialize window
        if user_id not in self.windows:
            self.windows[user_id] = FixedWindow(
                count=0,
                window_start=current_window_start
            )
        
        window = self.windows[user_id]
        
        # Check if we're in a new window
        if window.window_start < current_window_start:
            # New window started - reset counter
            window.count = 0
            window.window_start = current_window_start
        
        # Check if we can allow more requests
        if window.count < self.config.max_requests:
            # Increment counter and allow request
            window.count += 1
            
            remaining = self.config.max_requests - window.count
            
            return RateLimitResponse(
                allowed=True,
                remaining_requests=remaining,
                retry_after_seconds=None,
                message=f"Request allowed. {remaining} requests remaining in current window."
            )
        else:
            # Rate limit exceeded - calculate time until next window
            next_window_start = window.window_start + self.config.time_window_seconds
            retry_after = int(next_window_start - current_time) + 1
            
            return RateLimitResponse(
                allowed=False,
                remaining_requests=0,
                retry_after_seconds=retry_after,
                message=f"Rate limit exceeded. Retry after {retry_after} seconds (next window)."
            )
    
    def reset(self, user_id: str) -> None:
        """
        Reset rate limit for a user
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id in self.windows:
            del self.windows[user_id]


# ==================== FACTORY PATTERN ====================
# file: rate_limiter_factory.py (in production)

class RateLimiterFactory:
    """
    Factory class for creating rate limiter instances
    
    Factory Pattern Benefits:
    - Encapsulates object creation logic
    - Client code doesn't need to know concrete classes
    - Single point for validation and configuration
    - Easy to extend with new strategies
    - Follows Open/Closed Principle
    """
    
    @staticmethod
    def create(
        strategy: RateLimitStrategy,
        max_requests: int,
        time_window_seconds: int
    ) -> RateLimiter:
        """
        Create a rate limiter instance based on strategy
        
        Args:
            strategy: Rate limiting strategy (enum)
            max_requests: Maximum requests allowed
            time_window_seconds: Time window in seconds
            
        Returns:
            RateLimiter instance of the specified strategy
            
        Raises:
            ValueError: If strategy is unknown
            
        Example:
            limiter = RateLimiterFactory.create(
                RateLimitStrategy.TOKEN_BUCKET, 
                100, 
                60
            )
        """
        # Create configuration
        config = RateLimitConfig(max_requests, time_window_seconds)
        
        # Create appropriate rate limiter based on strategy
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowLogRateLimiter(config)
        
        elif strategy == RateLimitStrategy.FIXED_WINDOW:
            return FixedWindowCounterRateLimiter(config)
        
        else:
            raise ValueError(f"Unknown rate limit strategy: {strategy}")
    
    @staticmethod
    def create_from_string(
        strategy_name: str,
        max_requests: int,
        time_window_seconds: int
    ) -> RateLimiter:
        """
        Create a rate limiter from strategy name string
        Useful for configuration files or APIs
        
        Args:
            strategy_name: Strategy name as string
            max_requests: Maximum requests allowed
            time_window_seconds: Time window in seconds
            
        Returns:
            RateLimiter instance
            
        Raises:
            ValueError: If strategy name is invalid
            
        Example:
            limiter = RateLimiterFactory.create_from_string(
                "token_bucket", 
                100, 
                60
            )
        """
        try:
            strategy = RateLimitStrategy(strategy_name.lower())
            return RateLimiterFactory.create(strategy, max_requests, time_window_seconds)
        except ValueError:
            valid_strategies = [s.value for s in RateLimitStrategy]
            raise ValueError(
                f"Invalid strategy '{strategy_name}'. "
                f"Valid strategies: {', '.join(valid_strategies)}"
            )


# ==================== DEMO & TEST CODE ====================
# file: demo.py (in production)

def print_separator(title: str = "") -> None:
    """Print a visual separator for demo output"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}\n")


def demo_basic_usage() -> None:
    """Demonstrate basic usage of rate limiter"""
    print_separator("DEMO 1: Basic Usage")
    
    # Create a rate limiter: 3 requests per 10 seconds
    limiter = RateLimiterFactory.create(
        RateLimitStrategy.TOKEN_BUCKET,
        max_requests=3,
        time_window_seconds=10
    )
    
    user_id = "user123"
    
    print(f"Rate Limit: 3 requests per 10 seconds")
    print(f"Testing with user: {user_id}\n")
    
    # Make 5 requests
    for i in range(1, 6):
        response = limiter.allow_request(user_id)
        status = "✅ ALLOWED" if response.allowed else "❌ DENIED"
        print(f"Request {i}: {status}")
        print(f"  → {response.message}")
        if not response.allowed:
            print(f"  → Retry after: {response.retry_after_seconds}s")
        print()


def demo_multiple_users() -> None:
    """Demonstrate rate limiting with multiple users"""
    print_separator("DEMO 2: Multiple Users")
    
    limiter = RateLimiterFactory.create(
        RateLimitStrategy.SLIDING_WINDOW,
        max_requests=2,
        time_window_seconds=10
    )
    
    print(f"Rate Limit: 2 requests per 10 seconds")
    print(f"Testing with multiple users\n")
    
    users = ["alice", "bob", "alice", "bob", "alice"]
    
    for i, user in enumerate(users, 1):
        response = limiter.allow_request(user)
        status = "✅" if response.allowed else "❌"
        print(f"Request {i} ({user}): {status} {response.message}")


def demo_strategy_comparison() -> None:
    """Compare behavior of different strategies"""
    print_separator("DEMO 3: Strategy Comparison")
    
    # Create all three strategies with same config
    strategies = {
        "Token Bucket": RateLimiterFactory.create(
            RateLimitStrategy.TOKEN_BUCKET, 3, 5
        ),
        "Sliding Window": RateLimiterFactory.create(
            RateLimitStrategy.SLIDING_WINDOW, 3, 5
        ),
        "Fixed Window": RateLimiterFactory.create(
            RateLimitStrategy.FIXED_WINDOW, 3, 5
        ),
    }
    
    print("Rate Limit: 3 requests per 5 seconds")
    print("Making 4 rapid requests with each strategy:\n")
    
    for name, limiter in strategies.items():
        print(f"\n{name}:")
        allowed_count = 0
        for i in range(1, 5):
            response = limiter.allow_request("test_user")
            if response.allowed:
                allowed_count += 1
            status = "✅" if response.allowed else "❌"
            print(f"  Request {i}: {status}")
        print(f"  Total Allowed: {allowed_count}/4")


def demo_burst_traffic() -> None:
    """Demonstrate handling of burst traffic"""
    print_separator("DEMO 4: Burst Traffic Handling")
    
    limiter = RateLimiterFactory.create(
        RateLimitStrategy.TOKEN_BUCKET,
        max_requests=5,
        time_window_seconds=10
    )
    
    print("Token Bucket: 5 requests per 10 seconds")
    print("Simulating burst traffic:\n")
    
    # Burst of requests
    print("Burst - Making 7 rapid requests:")
    for i in range(1, 8):
        response = limiter.allow_request("burst_user")
        status = "✅" if response.allowed else "❌"
        remaining = response.remaining_requests if response.allowed else 0
        print(f"  Request {i}: {status} (Remaining: {remaining})")
    
    print("\nWaiting 3 seconds...")
    time.sleep(3)
    
    print("\nAfter 3 seconds - Tokens refilled:")
    response = limiter.allow_request("burst_user")
    print(f"  Request: {'✅' if response.allowed else '❌'}")
    print(f"  {response.message}")


def demo_edge_cases() -> None:
    """Demonstrate edge cases and error handling"""
    print_separator("DEMO 5: Edge Cases")
    
    print("1. Invalid Configuration:")
    try:
        config = RateLimitConfig(max_requests=-1, time_window_seconds=60)
    except ValueError as e:
        print(f"   ✅ Caught error: {e}\n")
    
    print("2. Invalid Strategy:")
    try:
        limiter = RateLimiterFactory.create_from_string("invalid_strategy", 100, 60)
    except ValueError as e:
        print(f"   ✅ Caught error: {e}\n")
    
    print("3. Reset Functionality:")
    limiter = RateLimiterFactory.create(RateLimitStrategy.FIXED_WINDOW, 2, 10)
    
    # Use up the limit
    limiter.allow_request("reset_user")
    limiter.allow_request("reset_user")
    response = limiter.allow_request("reset_user")
    print(f"   Before reset: {'✅' if response.allowed else '❌ DENIED'}")
    
    # Reset and try again
    limiter.reset("reset_user")
    response = limiter.allow_request("reset_user")
    print(f"   After reset: {'✅ ALLOWED' if response.allowed else '❌'}")


def demo_factory_pattern() -> None:
    """Demonstrate factory pattern usage"""
    print_separator("DEMO 6: Factory Pattern")
    
    print("Creating rate limiters using Factory Pattern:\n")
    
    # Method 1: Using Enum
    print("1. Create with Enum (Type-safe):")
    limiter1 = RateLimiterFactory.create(
        RateLimitStrategy.TOKEN_BUCKET,
        100,
        60
    )
    print(f"   ✅ Created: {limiter1.__class__.__name__}")
    
    # Method 2: Using String
    print("\n2. Create with String (Config-friendly):")
    limiter2 = RateLimiterFactory.create_from_string(
        "sliding_window",
        100,
        60
    )
    print(f"   ✅ Created: {limiter2.__class__.__name__}")
    
    # Different strategies for different use cases
    print("\n3. Different strategies for different endpoints:")
    configs = [
        ("Login API", "fixed_window", 5, 60),
        ("Public API", "token_bucket", 100, 60),
        ("Premium API", "sliding_window", 1000, 60),
    ]
    
    for name, strategy, max_req, window in configs:
        limiter = RateLimiterFactory.create_from_string(strategy, max_req, window)
        print(f"   {name}: {max_req} req/{window}s → {limiter.__class__.__name__}")


def run_all_demos() -> None:
    """Run all demonstration scenarios"""
    print("\n" + "="*60)
    print("  RATE LIMITER SYSTEM - DEMONSTRATION")
    print("="*60)
    
    demos = [
        demo_basic_usage,
        demo_multiple_users,
        demo_strategy_comparison,
        demo_burst_traffic,
        demo_edge_cases,
        demo_factory_pattern,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*60 + "\n")


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    In an interview, mention:
    - "Let me add some test code to validate the implementation"
    - "This demonstrates all key features and edge cases"
    - "In production, we'd use proper unit tests (pytest)"
    """
    run_all_demos()
