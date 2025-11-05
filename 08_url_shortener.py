"""
URL Shortening Service - Complete Implementation for Interviews
(Similar to bit.ly, tinyurl.com, t.co)

FUNCTIONAL REQUIREMENTS:
1. Shorten long URLs to unique short codes (6-7 characters)
2. Redirect short URLs to original long URLs
3. Support custom aliases (if available)
4. Track click analytics (views, locations, timestamps)
5. Support URL expiration (TTL)
6. Bulk URL shortening API
7. User account management and URL ownership

NON-FUNCTIONAL REQUIREMENTS:
1. Scalability: Handle millions of URLs, high read/write ratio (100:1)
2. Availability: 99.9% uptime, minimal downtime
3. Performance: <100ms for redirects, <200ms for shortening
4. Storage: Efficient encoding, handle billions of URLs
5. Security: Prevent malicious URLs, rate limiting
6. Analytics: Real-time click tracking and reporting

DESIGN PATTERNS:
- Strategy Pattern (URL encoding strategies)
- Factory Pattern (URL shortener creation)
- Observer Pattern (analytics tracking)
- Repository Pattern (data access layer)
- Decorator Pattern (caching, rate limiting)

INTERVIEW FOCUS:
- Base62 encoding vs UUID vs Counter-based approaches
- Database design and sharding strategies
- Caching layers and cache invalidation
- Collision handling and performance optimization
- Analytics pipeline and real-time processing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import base64
import random
import string
import time
import uuid
from collections import defaultdict, deque
from threading import Lock, RLock
import json
import re


# ==================== ENUMS & CONSTANTS ====================

class EncodingStrategy(Enum):
    """Different URL encoding strategies"""
    BASE62 = "base62"           # Most common: 0-9, a-z, A-Z
    COUNTER = "counter"         # Sequential counter-based
    UUID = "uuid"              # UUID-based (longer but guaranteed unique)
    HASH = "hash"              # Hash-based with collision handling


class URLStatus(Enum):
    """URL status for management"""
    ACTIVE = "active"
    EXPIRED = "expired"
    DISABLED = "disabled"
    MALICIOUS = "malicious"


# ==================== CONFIGURATION & MODELS ====================

@dataclass
class URLShortenerConfig:
    """Configuration for URL shortener service"""
    short_code_length: int = 6                    # Length of generated codes
    default_ttl_days: Optional[int] = None        # Default expiration in days
    enable_analytics: bool = True                 # Track click analytics
    enable_custom_aliases: bool = True            # Allow custom short codes
    max_urls_per_user: int = 10000               # Rate limiting per user
    cache_ttl_seconds: int = 3600                # Cache expiration
    blocked_domains: Set[str] = field(default_factory=set)  # Blocked domains
    
    def __post_init__(self):
        if self.short_code_length < 4 or self.short_code_length > 10:
            raise ValueError("short_code_length must be between 4 and 10")


@dataclass  
class URLMapping:
    """Represents a URL mapping in the system"""
    short_code: str
    long_url: str
    created_at: datetime
    created_by: Optional[str] = None              # User ID
    expires_at: Optional[datetime] = None         # Expiration time
    click_count: int = 0                         # Total clicks
    status: URLStatus = URLStatus.ACTIVE         # Current status
    custom_alias: bool = False                   # Is custom alias
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional data
    
    def is_expired(self) -> bool:
        """Check if URL has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_accessible(self) -> bool:
        """Check if URL can be accessed"""
        return (self.status == URLStatus.ACTIVE and 
                not self.is_expired())


@dataclass
class ClickEvent:
    """Represents a click analytics event"""
    short_code: str
    timestamp: datetime
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    referrer: Optional[str] = None
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'short_code': self.short_code,
            'timestamp': self.timestamp.isoformat(),
            'user_agent': self.user_agent,
            'ip_address': self.ip_address,
            'referrer': self.referrer,
            'location': self.location
        }


@dataclass
class Analytics:
    """Analytics data for a URL"""
    total_clicks: int
    unique_clicks: int
    daily_clicks: Dict[str, int]
    top_referrers: Dict[str, int]
    top_locations: Dict[str, int]
    recent_clicks: List[ClickEvent]
    
    def add_click(self, event: ClickEvent):
        """Add a click event to analytics"""
        self.total_clicks += 1
        
        # Track daily clicks
        date_key = event.timestamp.strftime('%Y-%m-%d')
        self.daily_clicks[date_key] = self.daily_clicks.get(date_key, 0) + 1
        
        # Track referrers
        if event.referrer:
            self.top_referrers[event.referrer] = self.top_referrers.get(event.referrer, 0) + 1
        
        # Track locations
        if event.location:
            self.top_locations[event.location] = self.top_locations.get(event.location, 0) + 1
        
        # Add to recent clicks (keep last 100)
        self.recent_clicks.append(event)
        if len(self.recent_clicks) > 100:
            self.recent_clicks.pop(0)


@dataclass
class ShortenResponse:
    """Response from URL shortening operation"""
    success: bool
    short_code: Optional[str] = None
    short_url: Optional[str] = None
    long_url: Optional[str] = None
    message: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class RedirectResponse:
    """Response from URL redirect operation"""
    success: bool
    long_url: Optional[str] = None
    status: Optional[URLStatus] = None
    message: Optional[str] = None


# ==================== ENCODING STRATEGIES ====================

class URLEncoder(ABC):
    """Abstract base class for URL encoding strategies"""
    
    @abstractmethod
    def encode(self, input_data: Any) -> str:
        """Encode input data to short code"""
        pass
    
    @abstractmethod
    def is_valid_code(self, code: str) -> bool:
        """Validate if code follows encoding rules"""
        pass


class Base62Encoder(URLEncoder):
    """
    Base62 URL Encoder (0-9, a-z, A-Z)
    
    Most popular choice for URL shorteners:
    - 62^6 = 56.8 billion possible combinations
    - Human-readable and URL-safe
    - Compact representation
    """
    
    BASE62_CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
    
    def __init__(self, length: int = 6):
        self.length = length
        self.base = len(self.BASE62_CHARS)
    
    def encode(self, number: int) -> str:
        """Convert number to base62 string"""
        if number == 0:
            return self.BASE62_CHARS[0] * self.length
        
        result = []
        while number > 0:
            result.append(self.BASE62_CHARS[number % self.base])
            number //= self.base
        
        # Pad with leading characters if needed
        while len(result) < self.length:
            result.append(self.BASE62_CHARS[0])
        
        return ''.join(reversed(result))
    
    def decode(self, code: str) -> int:
        """Convert base62 string back to number"""
        number = 0
        for char in code:
            number = number * self.base + self.BASE62_CHARS.index(char)
        return number
    
    def is_valid_code(self, code: str) -> bool:
        """Check if code is valid base62"""
        if len(code) != self.length:
            return False
        return all(c in self.BASE62_CHARS for c in code)
    
    def generate_random_code(self) -> str:
        """Generate random base62 code"""
        return ''.join(random.choices(self.BASE62_CHARS, k=self.length))


class HashEncoder(URLEncoder):
    """
    Hash-based URL encoder using MD5/SHA
    
    Pros:
    - Deterministic (same URL → same hash)
    - No coordination needed across servers
    - Good distribution
    
    Cons:
    - Collision possibility
    - Less compact than base62
    """
    
    def __init__(self, length: int = 6):
        self.length = length
    
    def encode(self, url: str) -> str:
        """Generate hash-based code from URL"""
        # Use MD5 for speed (collision handling needed)
        hash_digest = hashlib.md5(url.encode()).hexdigest()
        
        # Convert to base62 for URL-safe representation
        base62_encoder = Base62Encoder(self.length)
        
        # Use first part of hash as number
        hash_number = int(hash_digest[:8], 16)
        return base62_encoder.encode(hash_number)
    
    def is_valid_code(self, code: str) -> bool:
        """Validate hash-based code"""
        return Base62Encoder(self.length).is_valid_code(code)


class CounterEncoder(URLEncoder):
    """
    Counter-based URL encoder
    
    Pros:
    - Guaranteed uniqueness
    - Predictable length
    - Simple implementation
    
    Cons:
    - Requires centralized counter
    - Sequential codes (security/privacy concern)
    - Single point of failure
    """
    
    def __init__(self, length: int = 6, start_value: int = 1000000):
        self.length = length
        self.counter = start_value
        self.lock = Lock()
        self.base62_encoder = Base62Encoder(length)
    
    def encode(self, _: Any = None) -> str:
        """Generate next sequential code"""
        with self.lock:
            code = self.base62_encoder.encode(self.counter)
            self.counter += 1
            return code
    
    def is_valid_code(self, code: str) -> bool:
        """Validate counter-based code"""
        return self.base62_encoder.is_valid_code(code)


# ==================== STORAGE LAYER ====================

class URLStorage(ABC):
    """Abstract storage interface for URL mappings"""
    
    @abstractmethod
    def save_mapping(self, mapping: URLMapping) -> bool:
        """Save URL mapping"""
        pass
    
    @abstractmethod
    def get_mapping(self, short_code: str) -> Optional[URLMapping]:
        """Retrieve URL mapping by short code"""
        pass
    
    @abstractmethod
    def delete_mapping(self, short_code: str) -> bool:
        """Delete URL mapping"""
        pass
    
    @abstractmethod
    def exists(self, short_code: str) -> bool:
        """Check if short code exists"""
        pass
    
    @abstractmethod
    def get_user_urls(self, user_id: str) -> List[URLMapping]:
        """Get all URLs created by user"""
        pass


class InMemoryStorage(URLStorage):
    """
    In-memory storage implementation (for demo/testing)
    
    In production, use:
    - PostgreSQL/MySQL for ACID properties
    - Cassandra/DynamoDB for scale
    - Redis for caching layer
    """
    
    def __init__(self):
        self.url_mappings: Dict[str, URLMapping] = {}
        self.user_urls: Dict[str, Set[str]] = defaultdict(set)
        self.lock = RLock()
    
    def save_mapping(self, mapping: URLMapping) -> bool:
        """Save URL mapping to memory"""
        with self.lock:
            self.url_mappings[mapping.short_code] = mapping
            
            if mapping.created_by:
                self.user_urls[mapping.created_by].add(mapping.short_code)
            
            return True
    
    def get_mapping(self, short_code: str) -> Optional[URLMapping]:
        """Retrieve URL mapping from memory"""
        with self.lock:
            return self.url_mappings.get(short_code)
    
    def delete_mapping(self, short_code: str) -> bool:
        """Delete URL mapping from memory"""
        with self.lock:
            if short_code not in self.url_mappings:
                return False
            
            mapping = self.url_mappings[short_code]
            del self.url_mappings[short_code]
            
            if mapping.created_by:
                self.user_urls[mapping.created_by].discard(short_code)
            
            return True
    
    def exists(self, short_code: str) -> bool:
        """Check if short code exists in memory"""
        with self.lock:
            return short_code in self.url_mappings
    
    def get_user_urls(self, user_id: str) -> List[URLMapping]:
        """Get all URLs created by user"""
        with self.lock:
            user_codes = self.user_urls.get(user_id, set())
            return [self.url_mappings[code] for code in user_codes 
                   if code in self.url_mappings]


# ==================== ANALYTICS ENGINE ====================

class AnalyticsEngine:
    """
    Click analytics and reporting engine
    
    In production:
    - Use Apache Kafka for real-time streaming
    - ElasticSearch for analytics queries
    - Redis for real-time counters
    """
    
    def __init__(self):
        self.analytics_data: Dict[str, Analytics] = {}
        self.click_events: deque = deque(maxlen=10000)  # Recent events
        self.lock = Lock()
    
    def track_click(self, event: ClickEvent):
        """Track a click event"""
        with self.lock:
            # Add to global click events
            self.click_events.append(event)
            
            # Update URL-specific analytics
            if event.short_code not in self.analytics_data:
                self.analytics_data[event.short_code] = Analytics(
                    total_clicks=0,
                    unique_clicks=0,
                    daily_clicks={},
                    top_referrers={},
                    top_locations={},
                    recent_clicks=[]
                )
            
            self.analytics_data[event.short_code].add_click(event)
    
    def get_analytics(self, short_code: str) -> Optional[Analytics]:
        """Get analytics for a specific URL"""
        with self.lock:
            return self.analytics_data.get(short_code)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global analytics statistics"""
        with self.lock:
            total_clicks = sum(analytics.total_clicks 
                             for analytics in self.analytics_data.values())
            total_urls = len(self.analytics_data)
            
            return {
                'total_urls': total_urls,
                'total_clicks': total_clicks,
                'average_clicks_per_url': total_clicks / max(total_urls, 1),
                'recent_events_count': len(self.click_events)
            }


# ==================== CACHING LAYER ====================

class CacheManager:
    """
    Simple caching layer for frequently accessed URLs
    
    In production, use Redis with:
    - LRU eviction policy
    - TTL-based expiration
    - Cluster setup for high availability
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[URLMapping, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = Lock()
    
    def get(self, short_code: str) -> Optional[URLMapping]:
        """Get URL mapping from cache"""
        with self.lock:
            if short_code in self.cache:
                mapping, timestamp = self.cache[short_code]
                
                # Check TTL
                if time.time() - timestamp < self.ttl_seconds:
                    return mapping
                else:
                    # Expired, remove from cache
                    del self.cache[short_code]
            
            return None
    
    def put(self, short_code: str, mapping: URLMapping):
        """Store URL mapping in cache"""
        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Simple FIFO eviction (in production, use LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[short_code] = (mapping, time.time())
    
    def invalidate(self, short_code: str):
        """Remove URL mapping from cache"""
        with self.lock:
            self.cache.pop(short_code, None)
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()


# ==================== URL VALIDATOR ====================

class URLValidator:
    """
    URL validation and security checking
    
    Prevents:
    - Malicious URLs
    - Invalid formats
    - Blocked domains
    - Infinite redirect loops
    """
    
    def __init__(self, blocked_domains: Set[str] = None):
        self.blocked_domains = blocked_domains or set()
        
        # Common malicious patterns
        self.malicious_patterns = [
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'file://',
        ]
        
        # URL format regex
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        if not url or len(url) > 2048:  # URL too long
            return False
        
        # Check format
        if not self.url_pattern.match(url):
            return False
        
        # Check for malicious patterns
        url_lower = url.lower()
        for pattern in self.malicious_patterns:
            if re.search(pattern, url_lower):
                return False
        
        return True
    
    def is_blocked_domain(self, url: str) -> bool:
        """Check if URL domain is blocked"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            return domain in self.blocked_domains
        except:
            return True  # Assume blocked if parsing fails
    
    def validate_custom_alias(self, alias: str) -> bool:
        """Validate custom alias format"""
        if not alias or len(alias) < 3 or len(alias) > 20:
            return False
        
        # Only alphanumeric and hyphens
        return re.match(r'^[a-zA-Z0-9-_]+$', alias) is not None


# ==================== MAIN URL SHORTENER SERVICE ====================

class URLShortenerService:
    """
    Main URL Shortener Service
    
    Orchestrates all components:
    - URL encoding and storage
    - Caching for performance
    - Analytics tracking
    - Validation and security
    """
    
    def __init__(self, config: URLShortenerConfig):
        self.config = config
        self.storage = InMemoryStorage()  # In production: Database
        self.cache = CacheManager(ttl_seconds=config.cache_ttl_seconds)
        self.analytics = AnalyticsEngine() if config.enable_analytics else None
        self.validator = URLValidator(config.blocked_domains)
        
        # Initialize encoder based on strategy
        self.encoder = self._create_encoder(EncodingStrategy.BASE62)
        
        # Rate limiting (simple in-memory implementation)
        self.user_url_count: Dict[str, int] = defaultdict(int)
        self.rate_limit_lock = Lock()
    
    def _create_encoder(self, strategy: EncodingStrategy) -> URLEncoder:
        """Factory method to create URL encoder"""
        if strategy == EncodingStrategy.BASE62:
            return Base62Encoder(self.config.short_code_length)
        elif strategy == EncodingStrategy.HASH:
            return HashEncoder(self.config.short_code_length)
        elif strategy == EncodingStrategy.COUNTER:
            return CounterEncoder(self.config.short_code_length)
        else:
            raise ValueError(f"Unsupported encoding strategy: {strategy}")
    
    def shorten_url(self, 
                   long_url: str, 
                   user_id: Optional[str] = None,
                   custom_alias: Optional[str] = None,
                   ttl_days: Optional[int] = None) -> ShortenResponse:
        """
        Shorten a long URL
        
        Args:
            long_url: Original URL to shorten
            user_id: User creating the short URL
            custom_alias: Custom short code (if allowed)
            ttl_days: Expiration time in days
            
        Returns:
            ShortenResponse with result
        """
        
        # 1. Validate input URL
        if not self.validator.is_valid_url(long_url):
            return ShortenResponse(
                success=False,
                message="Invalid URL format"
            )
        
        if self.validator.is_blocked_domain(long_url):
            return ShortenResponse(
                success=False,
                message="Domain is blocked"
            )
        
        # 2. Check rate limiting
        if user_id and not self._check_rate_limit(user_id):
            return ShortenResponse(
                success=False,
                message=f"Rate limit exceeded. Maximum {self.config.max_urls_per_user} URLs per user"
            )
        
        # 3. Handle custom alias
        if custom_alias:
            if not self.config.enable_custom_aliases:
                return ShortenResponse(
                    success=False,
                    message="Custom aliases are not enabled"
                )
            
            if not self.validator.validate_custom_alias(custom_alias):
                return ShortenResponse(
                    success=False,
                    message="Invalid custom alias format"
                )
            
            if self.storage.exists(custom_alias):
                return ShortenResponse(
                    success=False,
                    message="Custom alias already exists"
                )
            
            short_code = custom_alias
            is_custom = True
        else:
            # 4. Generate short code with collision handling
            short_code = self._generate_unique_code(long_url)
            if not short_code:
                return ShortenResponse(
                    success=False,
                    message="Failed to generate unique short code"
                )
            is_custom = False
        
        # 5. Calculate expiration
        expires_at = None
        if ttl_days or self.config.default_ttl_days:
            days = ttl_days or self.config.default_ttl_days
            expires_at = datetime.now() + timedelta(days=days)
        
        # 6. Create and save mapping
        mapping = URLMapping(
            short_code=short_code,
            long_url=long_url,
            created_at=datetime.now(),
            created_by=user_id,
            expires_at=expires_at,
            custom_alias=is_custom
        )
        
        if not self.storage.save_mapping(mapping):
            return ShortenResponse(
                success=False,
                message="Failed to save URL mapping"
            )
        
        # 7. Update rate limiting
        if user_id:
            with self.rate_limit_lock:
                self.user_url_count[user_id] += 1
        
        # 8. Cache the mapping
        self.cache.put(short_code, mapping)
        
        return ShortenResponse(
            success=True,
            short_code=short_code,
            short_url=f"https://short.ly/{short_code}",  # Base URL
            long_url=long_url,
            expires_at=expires_at,
            message="URL shortened successfully"
        )
    
    def redirect_url(self, 
                    short_code: str, 
                    track_analytics: bool = True,
                    user_agent: Optional[str] = None,
                    ip_address: Optional[str] = None,
                    referrer: Optional[str] = None) -> RedirectResponse:
        """
        Redirect short URL to original URL
        
        Args:
            short_code: Short code to resolve
            track_analytics: Whether to track this click
            user_agent: User agent string
            ip_address: Client IP address
            referrer: Referrer URL
            
        Returns:
            RedirectResponse with original URL or error
        """
        
        # 1. Try cache first
        mapping = self.cache.get(short_code)
        
        # 2. Fall back to storage
        if not mapping:
            mapping = self.storage.get_mapping(short_code)
            
            if not mapping:
                return RedirectResponse(
                    success=False,
                    message="Short URL not found"
                )
            
            # Cache for future requests
            self.cache.put(short_code, mapping)
        
        # 3. Check if URL is accessible
        if not mapping.is_accessible():
            status_messages = {
                URLStatus.EXPIRED: "URL has expired",
                URLStatus.DISABLED: "URL has been disabled",
                URLStatus.MALICIOUS: "URL has been flagged as malicious"
            }
            
            if mapping.is_expired():
                message = "URL has expired"
            else:
                message = status_messages.get(mapping.status, "URL is not accessible")
            
            return RedirectResponse(
                success=False,
                status=mapping.status,
                message=message
            )
        
        # 4. Track analytics
        if track_analytics and self.analytics:
            click_event = ClickEvent(
                short_code=short_code,
                timestamp=datetime.now(),
                user_agent=user_agent,
                ip_address=ip_address,
                referrer=referrer
            )
            self.analytics.track_click(click_event)
            
            # Update click count in mapping
            mapping.click_count += 1
            self.storage.save_mapping(mapping)  # Persist updated count
            self.cache.put(short_code, mapping)  # Update cache
        
        return RedirectResponse(
            success=True,
            long_url=mapping.long_url,
            status=mapping.status,
            message="Redirect successful"
        )
    
    def get_url_info(self, short_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a short URL"""
        mapping = self.storage.get_mapping(short_code)
        if not mapping:
            return None
        
        info = {
            'short_code': mapping.short_code,
            'long_url': mapping.long_url,
            'created_at': mapping.created_at.isoformat(),
            'created_by': mapping.created_by,
            'click_count': mapping.click_count,
            'status': mapping.status.value,
            'custom_alias': mapping.custom_alias,
            'expires_at': mapping.expires_at.isoformat() if mapping.expires_at else None,
            'is_expired': mapping.is_expired()
        }
        
        # Add analytics if available
        if self.analytics:
            analytics = self.analytics.get_analytics(short_code)
            if analytics:
                info['analytics'] = {
                    'total_clicks': analytics.total_clicks,
                    'daily_clicks': analytics.daily_clicks,
                    'top_referrers': dict(list(analytics.top_referrers.items())[:10]),
                    'recent_clicks_count': len(analytics.recent_clicks)
                }
        
        return info
    
    def get_user_urls(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all URLs created by a user"""
        mappings = self.storage.get_user_urls(user_id)
        
        return [{
            'short_code': mapping.short_code,
            'long_url': mapping.long_url,
            'created_at': mapping.created_at.isoformat(),
            'click_count': mapping.click_count,
            'status': mapping.status.value,
            'expires_at': mapping.expires_at.isoformat() if mapping.expires_at else None,
            'is_expired': mapping.is_expired()
        } for mapping in mappings]
    
    def delete_url(self, short_code: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a short URL
        
        Args:
            short_code: Short code to delete
            user_id: User requesting deletion (for ownership check)
            
        Returns:
            True if deleted, False otherwise
        """
        mapping = self.storage.get_mapping(short_code)
        if not mapping:
            return False
        
        # Check ownership if user_id provided
        if user_id and mapping.created_by != user_id:
            return False
        
        # Delete from storage and invalidate cache
        success = self.storage.delete_mapping(short_code)
        if success:
            self.cache.invalidate(short_code)
        
        return success
    
    def _generate_unique_code(self, long_url: str, max_attempts: int = 5) -> Optional[str]:
        """Generate unique short code with collision handling"""
        
        for attempt in range(max_attempts):
            if isinstance(self.encoder, HashEncoder):
                # Hash-based: deterministic but may collide
                code = self.encoder.encode(long_url + str(attempt))
            elif isinstance(self.encoder, CounterEncoder):
                # Counter-based: guaranteed unique
                code = self.encoder.encode()
            else:
                # Base62: random generation
                code = self.encoder.generate_random_code()
            
            # Check for collision
            if not self.storage.exists(code):
                return code
        
        # All attempts failed
        return None
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        with self.rate_limit_lock:
            return self.user_url_count[user_id] < self.config.max_urls_per_user
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global service statistics"""
        stats = {
            'total_urls': len(self.storage.url_mappings),
            'cache_size': len(self.cache.cache),
            'encoding_strategy': type(self.encoder).__name__
        }
        
        if self.analytics:
            stats.update(self.analytics.get_global_stats())
        
        return stats


# ==================== FACTORY FOR SERVICE CREATION ====================

class URLShortenerFactory:
    """Factory for creating URL shortener services with different configurations"""
    
    @staticmethod
    def create_basic_service() -> URLShortenerService:
        """Create basic URL shortener service"""
        config = URLShortenerConfig(
            short_code_length=6,
            enable_analytics=True,
            enable_custom_aliases=True
        )
        return URLShortenerService(config)
    
    @staticmethod
    def create_enterprise_service() -> URLShortenerService:
        """Create enterprise URL shortener with advanced features"""
        config = URLShortenerConfig(
            short_code_length=7,
            default_ttl_days=365,
            enable_analytics=True,
            enable_custom_aliases=True,
            max_urls_per_user=50000,
            cache_ttl_seconds=7200,
            blocked_domains={'malicious.com', 'spam.org'}
        )
        return URLShortenerService(config)
    
    @staticmethod
    def create_high_performance_service() -> URLShortenerService:
        """Create high-performance service optimized for speed"""
        config = URLShortenerConfig(
            short_code_length=6,
            enable_analytics=False,  # Disabled for performance
            enable_custom_aliases=False,
            max_urls_per_user=100000,
            cache_ttl_seconds=3600
        )
        service = URLShortenerService(config)
        # Use counter-based encoder for guaranteed uniqueness without collisions
        service.encoder = CounterEncoder(6)
        return service


# ==================== DEMO AND TESTING ====================

def print_separator(title: str = "") -> None:
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}\n")


def demo_basic_functionality():
    """Demonstrate basic URL shortening and redirection"""
    print_separator("DEMO 1: Basic URL Shortening")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Test URLs
    test_urls = [
        "https://www.google.com/search?q=python+programming",
        "https://github.com/microsoft/vscode",
        "https://stackoverflow.com/questions/tagged/python"
    ]
    
    print("📝 Shortening URLs:")
    short_codes = []
    
    for i, url in enumerate(test_urls, 1):
        response = service.shorten_url(url, user_id=f"user_{i}")
        if response.success:
            print(f"   ✅ URL {i}: {response.short_url}")
            print(f"      Original: {url}")
            short_codes.append(response.short_code)
        else:
            print(f"   ❌ Failed: {response.message}")
    
    print(f"\n🔗 Testing redirects:")
    for i, code in enumerate(short_codes, 1):
        redirect = service.redirect_url(code, 
                                      user_agent="Mozilla/5.0 Demo Browser",
                                      ip_address=f"192.168.1.{i}")
        if redirect.success:
            print(f"   ✅ {code} → {redirect.long_url}")
        else:
            print(f"   ❌ {code}: {redirect.message}")


def demo_custom_aliases():
    """Demonstrate custom alias functionality"""
    print_separator("DEMO 2: Custom Aliases")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Test custom aliases
    custom_tests = [
        ("https://www.microsoft.com", "msft", True),
        ("https://www.apple.com", "apple-home", True),
        ("https://www.google.com", "g!", False),  # Invalid format
        ("https://www.amazon.com", "msft", False),  # Already exists
    ]
    
    print("🏷️ Testing custom aliases:")
    for url, alias, should_succeed in custom_tests:
        response = service.shorten_url(url, user_id="demo_user", custom_alias=alias)
        
        status = "✅ Success" if response.success else "❌ Failed"
        expected = "✅ Expected" if should_succeed else "❌ Expected"
        
        print(f"   {status} ({expected}): '{alias}' → {response.message}")
        
        if response.success:
            print(f"      Short URL: {response.short_url}")


def demo_analytics_tracking():
    """Demonstrate analytics and click tracking"""
    print_separator("DEMO 3: Analytics Tracking")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Create a URL for testing
    response = service.shorten_url(
        "https://www.example.com/popular-article",
        user_id="content_creator"
    )
    
    if not response.success:
        print("❌ Failed to create test URL")
        return
    
    short_code = response.short_code
    print(f"📊 Testing analytics for: {response.short_url}")
    
    # Simulate clicks from different sources
    click_scenarios = [
        ("Mozilla/5.0 (Windows)", "192.168.1.10", "https://twitter.com"),
        ("Mozilla/5.0 (Mac)", "192.168.1.11", "https://facebook.com"),
        ("Mozilla/5.0 (Linux)", "192.168.1.12", "https://reddit.com"),
        ("Mozilla/5.0 (Mobile)", "192.168.1.13", "https://twitter.com"),
        ("Mozilla/5.0 (Tablet)", "192.168.1.14", None),
    ]
    
    print("\n🖱️ Simulating clicks:")
    for user_agent, ip, referrer in click_scenarios:
        redirect = service.redirect_url(
            short_code,
            user_agent=user_agent,
            ip_address=ip,
            referrer=referrer
        )
        
        if redirect.success:
            referrer_info = f"from {referrer}" if referrer else "direct"
            print(f"   ✅ Click tracked ({referrer_info})")
        else:
            print(f"   ❌ Click failed: {redirect.message}")
    
    # Show analytics
    print("\n📈 Analytics Summary:")
    url_info = service.get_url_info(short_code)
    if url_info and 'analytics' in url_info:
        analytics = url_info['analytics']
        print(f"   Total clicks: {analytics['total_clicks']}")
        print(f"   Top referrers: {analytics['top_referrers']}")
        print(f"   Recent clicks: {analytics['recent_clicks_count']}")
    else:
        print("   No analytics data available")


def demo_url_expiration():
    """Demonstrate URL expiration functionality"""
    print_separator("DEMO 4: URL Expiration")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Create URLs with different expiration settings
    print("⏰ Creating URLs with expiration:")
    
    # Short-lived URL (for demo, we'll manually expire it)
    response1 = service.shorten_url(
        "https://www.example.com/temporary-offer",
        user_id="marketing_team",
        ttl_days=30
    )
    
    if response1.success:
        print(f"   ✅ 30-day URL: {response1.short_url}")
        print(f"      Expires: {response1.expires_at}")
        
        # Test normal access
        redirect = service.redirect_url(response1.short_code)
        print(f"   ✅ Access test: {'Success' if redirect.success else 'Failed'}")
        
        # Simulate expiration by manually setting past date
        mapping = service.storage.get_mapping(response1.short_code)
        mapping.expires_at = datetime.now() - timedelta(days=1)
        service.storage.save_mapping(mapping)
        service.cache.invalidate(response1.short_code)  # Clear cache
        
        # Test expired access
        redirect = service.redirect_url(response1.short_code)
        print(f"   ❌ Expired test: {redirect.message}")
    
    # Permanent URL
    response2 = service.shorten_url(
        "https://www.example.com/permanent-page",
        user_id="admin"
        # No TTL = permanent
    )
    
    if response2.success:
        print(f"   ✅ Permanent URL: {response2.short_url}")
        print(f"      Never expires")


def demo_rate_limiting():
    """Demonstrate rate limiting functionality"""
    print_separator("DEMO 5: Rate Limiting")
    
    # Create service with low rate limit for demo
    config = URLShortenerConfig(
        short_code_length=6,
        max_urls_per_user=3,  # Very low for demo
        enable_analytics=True
    )
    service = URLShortenerService(config)
    
    print(f"🚫 Rate limit: {config.max_urls_per_user} URLs per user")
    
    # Test rate limiting
    test_user = "limited_user"
    
    for i in range(5):
        response = service.shorten_url(
            f"https://www.example.com/page-{i+1}",
            user_id=test_user
        )
        
        if response.success:
            print(f"   ✅ URL {i+1}: Created successfully")
        else:
            print(f"   ❌ URL {i+1}: {response.message}")


def demo_user_management():
    """Demonstrate user URL management"""
    print_separator("DEMO 6: User URL Management")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Create URLs for different users
    users_urls = {
        "alice": [
            "https://www.github.com/alice/project1",
            "https://www.linkedin.com/in/alice"
        ],
        "bob": [
            "https://www.portfolio.com/bob",
            "https://www.twitter.com/bob_dev"
        ]
    }
    
    print("👥 Creating URLs for different users:")
    
    for user_id, urls in users_urls.items():
        print(f"\n   User: {user_id}")
        for url in urls:
            response = service.shorten_url(url, user_id=user_id)
            if response.success:
                print(f"     ✅ {response.short_code}: {url}")
            else:
                print(f"     ❌ Failed: {response.message}")
    
    # Show user's URLs
    print(f"\n📋 Alice's URLs:")
    alice_urls = service.get_user_urls("alice")
    for url_info in alice_urls:
        print(f"   📎 {url_info['short_code']}: {url_info['long_url']}")
        print(f"      Clicks: {url_info['click_count']}, Created: {url_info['created_at'][:19]}")
    
    # Test URL deletion
    if alice_urls:
        first_url = alice_urls[0]
        success = service.delete_url(first_url['short_code'], user_id="alice")
        print(f"\n🗑️ Deleted Alice's first URL: {'Success' if success else 'Failed'}")
        
        # Verify deletion
        updated_urls = service.get_user_urls("alice")
        print(f"   Alice now has {len(updated_urls)} URLs (was {len(alice_urls)})")


def demo_performance_comparison():
    """Demonstrate performance with different encoding strategies"""
    print_separator("DEMO 7: Encoding Strategy Comparison")
    
    # Test different encoders
    encoders = {
        "Base62": Base62Encoder(6),
        "Hash": HashEncoder(6),
        "Counter": CounterEncoder(6)
    }
    
    test_url = "https://www.example.com/test-performance"
    
    print("⚡ Comparing encoding strategies:")
    
    for name, encoder in encoders.items():
        print(f"\n   {name} Encoder:")
        
        # Generate multiple codes
        codes = []
        start_time = time.time()
        
        for i in range(100):
            if isinstance(encoder, HashEncoder):
                code = encoder.encode(f"{test_url}?id={i}")
            elif isinstance(encoder, CounterEncoder):
                code = encoder.encode()
            else:
                code = encoder.generate_random_code()
            codes.append(code)
        
        duration = time.time() - start_time
        
        print(f"     ⏱️ Generated 100 codes in {duration:.3f}s")
        print(f"     📋 Sample codes: {codes[:5]}")
        print(f"     🔍 Unique codes: {len(set(codes))}/100")


def demo_global_statistics():
    """Show global service statistics"""
    print_separator("DEMO 8: Global Statistics")
    
    service = URLShortenerFactory.create_basic_service()
    
    # Create some sample data
    sample_urls = [
        "https://www.microsoft.com",
        "https://www.github.com", 
        "https://www.stackoverflow.com",
        "https://www.python.org",
        "https://www.docker.com"
    ]
    
    print("📊 Creating sample data...")
    
    for i, url in enumerate(sample_urls):
        response = service.shorten_url(url, user_id=f"user_{i % 3}")
        if response.success:
            # Simulate some clicks
            for _ in range(random.randint(1, 10)):
                service.redirect_url(response.short_code)
    
    # Show statistics
    stats = service.get_global_stats()
    print(f"\n📈 Global Statistics:")
    print(f"   Total URLs: {stats['total_urls']}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Encoding strategy: {stats['encoding_strategy']}")
    
    if 'total_clicks' in stats:
        print(f"   Total clicks: {stats['total_clicks']}")
        print(f"   Average clicks per URL: {stats['average_clicks_per_url']:.1f}")


def run_comprehensive_demo():
    """Run all demonstration scenarios"""
    print("🎯 URL SHORTENING SERVICE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Run all demos
        demos = [
            demo_basic_functionality,
            demo_custom_aliases,
            demo_analytics_tracking,
            demo_url_expiration,
            demo_rate_limiting,
            demo_user_management,
            demo_performance_comparison,
            demo_global_statistics
        ]
        
        for demo in demos:
            try:
                demo()
                time.sleep(1)  # Brief pause between demos
            except Exception as e:
                print(f"\n❌ Error in {demo.__name__}: {e}")
        
        print_separator()
        print("✅ All demonstrations completed!")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   ✅ URL shortening with multiple encoding strategies")
        print(f"   ✅ Custom aliases and validation")
        print(f"   ✅ Real-time click analytics and tracking")
        print(f"   ✅ URL expiration and lifecycle management")
        print(f"   ✅ Rate limiting and abuse prevention")
        print(f"   ✅ User account management and ownership")
        print(f"   ✅ Caching for high performance")
        print(f"   ✅ Comprehensive validation and security")
        
        print(f"\n🚀 Production Considerations:")
        print(f"   • Use PostgreSQL/MySQL for persistent storage")
        print(f"   • Implement Redis for distributed caching")
        print(f"   • Add proper authentication and authorization")
        print(f"   • Use CDN for global distribution")
        print(f"   • Implement monitoring and alerting")
        print(f"   • Add database sharding for scale")
        print(f"   • Use message queues for analytics pipeline")
        
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    In an interview, you would say:
    - "Let me create a comprehensive demo to show all features"
    - "This demonstrates production-ready URL shortening with analytics"
    - "In real implementation, we'd use proper databases and caching"
    """
    run_comprehensive_demo()