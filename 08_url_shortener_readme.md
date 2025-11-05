# URL Shortening Service - Interview Guide

## 📋 Overview
A production-ready URL shortening service (like bit.ly, tinyurl.com) implementing multiple encoding strategies, analytics tracking, caching, and comprehensive security features.

## 🎯 Interview Focus Areas

### Core Concepts to Master
1. **URL Encoding Strategies**: Base62, Hash-based, Counter-based approaches
2. **Database Design**: Schema design, indexing, sharding strategies
3. **Caching Architecture**: Multi-level caching, cache invalidation
4. **Analytics Pipeline**: Real-time click tracking and reporting
5. **System Scalability**: Load balancing, distributed architecture
6. **Security**: URL validation, rate limiting, abuse prevention

## 🔥 Step-by-Step Implementation Guide

### Phase 1: Requirements Clarification (3-4 minutes)
**Essential questions to ask:**
```
Q: What's the expected scale? (URLs per day, requests per second)
Q: Do we need custom aliases support?
Q: Should we track analytics (clicks, referrers, locations)?
Q: Do URLs need expiration/TTL support?
Q: How long should the short codes be?
Q: Do we need user accounts and ownership?
Q: What about malicious URL protection?
Q: Are there any specific domains we should block?
```

### Phase 2: High-Level System Design (4-5 minutes)
```
    Client/Browser
         ↓
┌─────────────────────────────────────────────┐
│              Load Balancer                  │
└─────────────────────────────────────────────┘
         ↓                    ↓
┌─────────────────┐    ┌─────────────────┐
│  Shortening API │    │  Redirect API   │
│   (Write-Heavy) │    │  (Read-Heavy)   │
└─────────────────┘    └─────────────────┘
         ↓                    ↓
┌─────────────────────────────────────────────┐
│              Cache Layer (Redis)            │
└─────────────────────────────────────────────┘
         ↓                    ↓
┌─────────────────┐    ┌─────────────────┐
│   URL Database  │    │ Analytics Store │
│  (PostgreSQL/   │    │   (ClickHouse/  │
│   Cassandra)    │    │   ElasticSearch)│
└─────────────────┘    └─────────────────┘
```

### Phase 3: Database Schema Design (5-6 minutes)

#### URL Mappings Table
```sql
CREATE TABLE url_mappings (
    short_code VARCHAR(10) PRIMARY KEY,
    long_url TEXT NOT NULL,
    user_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NULL,
    click_count BIGINT DEFAULT 0,
    status ENUM('active', 'expired', 'disabled', 'malicious') DEFAULT 'active',
    custom_alias BOOLEAN DEFAULT false,
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_expires_at (expires_at)
);
```

#### Analytics Events Table  
```sql
CREATE TABLE click_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    short_code VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address VARCHAR(45),
    user_agent TEXT,
    referrer TEXT,
    location VARCHAR(100),
    
    INDEX idx_short_code_timestamp (short_code, timestamp),
    INDEX idx_timestamp (timestamp)
);
```

### Phase 4: URL Encoding Implementation (8-10 minutes)

#### Start with Base62 Encoding (Most Popular)
```python
class Base62Encoder:
    BASE62_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def encode(self, number: int) -> str:
        """Convert number to base62 string"""
        if number == 0:
            return self.BASE62_CHARS[0]
        
        result = []
        base = len(self.BASE62_CHARS)
        
        while number > 0:
            result.append(self.BASE62_CHARS[number % base])
            number //= base
        
        return ''.join(reversed(result))
    
    def decode(self, code: str) -> int:
        """Convert base62 string back to number"""
        number = 0
        base = len(self.BASE62_CHARS)
        
        for char in code:
            number = number * base + self.BASE62_CHARS.index(char)
        
        return number
```

**🎯 Key Implementation Points:**
1. **Base62 choice**: 62^6 = 56.8 billion combinations (enough for most use cases)
2. **Collision handling**: Use counter-based approach or retry with different input
3. **URL validation**: Regex patterns, domain blacklists
4. **Custom aliases**: Separate validation and availability checking

## 📚 Encoding Strategy Deep Dive

### 1. Base62 Encoding (Recommended)
```python
# Advantages:
# - Human-readable: Only letters and numbers
# - URL-safe: No special characters needing encoding  
# - Compact: 6 chars = 56.8 billion combinations
# - Predictable length: Always same number of characters

# Example: 123456789 → "B7E4y9"
```

**Pros:**
- Most compact representation
- URL-safe without encoding
- Human-readable and shareable
- Industry standard (used by bit.ly, t.co)

**Cons:**
- Need collision handling for random generation
- Slightly more complex than simple hash

### 2. Counter-Based Encoding
```python
# Use auto-incrementing counter as base for encoding
# Example: Counter 1000000 → Base62 → "4c92"

class CounterBasedEncoder:
    def __init__(self):
        self.counter = 1000000  # Start with larger number
        self.lock = threading.Lock()
    
    def generate_code(self) -> str:
        with self.lock:
            code = self.base62_encode(self.counter)
            self.counter += 1
            return code
```

**Pros:**
- Guaranteed uniqueness (no collisions)
- Predictable performance O(1)
- Simple implementation
- Works well in single-datacenter setup

**Cons:**
- Sequential codes (security/privacy concern)
- Single point of failure (counter)
- Difficult to distribute across multiple servers
- Can reveal business metrics (creation rate)

### 3. Hash-Based Encoding
```python
# Hash the URL and take first N characters
# Example: MD5("https://google.com")[:6] → "a9b8c7"

import hashlib

def hash_encode(url: str, length: int = 6) -> str:
    hash_digest = hashlib.md5(url.encode()).hexdigest()
    return hash_digest[:length]
```

**Pros:**
- Deterministic (same URL → same short code)
- No coordination needed between servers
- Good for caching (same URL always maps to same code)

**Cons:**
- Higher collision probability
- Fixed length may not provide enough uniqueness
- Need robust collision handling
- Potential for malicious collision attacks

### 4. UUID-Based Encoding
```python
# Generate UUID and encode to base62
# Longer but guaranteed unique

import uuid

def uuid_encode() -> str:
    return base62_encode(uuid.uuid4().int)[:8]  # Take first 8 chars
```

**Pros:**
- Guaranteed global uniqueness
- No coordination required
- Cryptographically random

**Cons:**
- Longer codes (usually 8+ characters)
- Less user-friendly
- Overkill for most use cases

## ⚡ System Design Considerations

### Scalability Architecture

#### 1. Read vs Write Patterns
```
Read:Write Ratio = 100:1 (typical for URL shorteners)

Write Path (URL Creation):
Client → Load Balancer → App Server → Database
                    ↓
                Cache Write (async)

Read Path (URL Redirect):  
Client → Load Balancer → Cache → App Server → Database (if cache miss)
```

#### 2. Database Sharding Strategy
```python
# Shard by short_code for even distribution
def get_shard_id(short_code: str, num_shards: int) -> int:
    return hash(short_code) % num_shards

# Example with 4 shards:
# "abc123" → Shard 2
# "xyz789" → Shard 1
```

#### 3. Caching Strategy
```
L1 Cache: Application-level (LRU, 1000 entries)
L2 Cache: Redis cluster (100M entries, 1-hour TTL)
L3 Cache: CDN (geographical distribution)

Cache Key Pattern: "url:{short_code}"
Cache Value: JSON with long_url, expires_at, etc.
```

### Performance Optimizations

#### 1. Database Indexing
```sql
-- Primary access pattern: lookup by short_code
PRIMARY KEY (short_code)

-- User management
INDEX idx_user_created (user_id, created_at DESC)

-- Analytics queries
INDEX idx_clicks_date (short_code, created_at)

-- Cleanup expired URLs
INDEX idx_expires_at (expires_at) WHERE expires_at IS NOT NULL
```

#### 2. Connection Pooling
```python
# Database connection pool
DATABASE_POOL = {
    'min_connections': 5,
    'max_connections': 20,
    'connection_timeout': 30
}

# Redis connection pool
REDIS_POOL = {
    'max_connections': 50,
    'retry_on_timeout': True
}
```

#### 3. Async Processing
```python
# Analytics processing (non-blocking)
async def track_click_async(event: ClickEvent):
    await analytics_queue.put(event)
    # Process in background worker

# Batch analytics inserts
async def process_analytics_batch():
    events = await analytics_queue.get_batch(size=1000)
    await database.bulk_insert(events)
```

## 🎯 Do's and Don'ts

### ✅ DO's
1. **Start with requirements**: Clarify scale, features, constraints
2. **Choose appropriate encoding**: Base62 for most cases, counter for guaranteed uniqueness
3. **Design for read-heavy workload**: Heavy caching, read replicas
4. **Handle edge cases**: URL validation, collision resolution, rate limiting  
5. **Plan for analytics**: Separate storage, async processing
6. **Consider security**: Input validation, domain blacklists, rate limiting
7. **Design for failure**: Circuit breakers, graceful degradation

### ❌ DON'Ts  
1. **Don't ignore collision handling**: Always have a strategy for duplicates
2. **Don't store everything in one database**: Separate analytics from core data
3. **Don't forget about cache invalidation**: Handle URL updates/deletes
4. **Don't make codes too short**: Balance between length and collision probability
5. **Don't ignore malicious URLs**: Validate and filter dangerous content
6. **Don't over-engineer initially**: Start simple, add complexity as needed
7. **Don't forget monitoring**: Track error rates, latencies, cache hit rates

## 🎤 Expected Interview Questions & Answers

### Q1: "How do you handle high read traffic (millions of redirects per second)?"
**A**: "Multi-layered caching strategy:

1. **CDN Layer**: Cache popular URLs at edge locations
   - 90%+ cache hit rate globally
   - Geographic distribution reduces latency

2. **Application Cache**: In-memory LRU cache
   - 1000-10000 most recent URLs
   - Sub-millisecond access time

3. **Redis Cluster**: Distributed cache
   - 100M+ URLs cached with 1-hour TTL
   - Handles cache misses from application layer

4. **Database Read Replicas**: Multiple read-only databases
   - Handle cache misses
   - Load balanced across replicas

5. **Preloading**: Async preload popular URLs into cache"

### Q2: "How do you ensure uniqueness across multiple servers generating short codes?"
**A**: "Several approaches depending on consistency requirements:

1. **Counter-Based with Ranges**:
   ```python
   # Each server gets range of counters
   Server 1: 1,000,000 - 1,999,999
   Server 2: 2,000,000 - 2,999,999
   # Coordinate through database or service discovery
   ```

2. **UUID + Base62**: 
   - Generate UUID, convert to base62
   - Longer codes but guaranteed unique
   - No coordination needed

3. **Hash + Collision Handling**:
   ```python
   def generate_with_retry(url, max_attempts=5):
       for i in range(max_attempts):
           code = hash_encode(url + str(i))
           if not exists_in_database(code):
               return code
       raise Exception("Failed to generate unique code")
   ```

4. **Centralized Counter Service**:
   - Dedicated service distributes counters
   - High availability with leader election
   - Batch allocation for performance"

### Q3: "How do you design the analytics system for billions of clicks?"
**A**: "Event-driven architecture with batch processing:

1. **Real-time Ingestion**:
   ```python
   # Fast write path - no blocking operations
   async def track_click(event):
       await kafka_producer.send('clicks', event)
       await redis.incr(f'counter:{short_code}')  # Real-time counter
   ```

2. **Stream Processing** (Apache Kafka + Spark/Flink):
   - Process click events in real-time
   - Aggregate metrics (daily/hourly clicks)
   - Detect patterns and anomalies

3. **Analytics Storage**:
   - **Hot data**: Redis for real-time counters
   - **Warm data**: ClickHouse for recent analytics queries  
   - **Cold data**: S3/Hadoop for long-term storage

4. **Query Layer**:
   ```python
   # API returns pre-aggregated data
   def get_analytics(short_code, time_range):
       # Check Redis for real-time data
       # Query ClickHouse for historical data
       # Combine and return
   ```

5. **Batch Jobs**: Nightly jobs for complex analytics
   - Top referrers, geographic distribution
   - User behavior analysis
   - Data archival and cleanup"

### Q4: "How do you handle URL expiration at scale?"
**A**: "Lazy deletion with background cleanup:

1. **Lazy Expiration Check**:
   ```python
   def redirect_url(short_code):
       mapping = get_from_cache_or_db(short_code)
       if mapping.is_expired():
           # Mark as expired, don't redirect
           return expired_response()
       return redirect(mapping.long_url)
   ```

2. **Background Cleanup Jobs**:
   ```python
   # Scheduled job runs every hour
   def cleanup_expired_urls():
       expired_urls = db.query(
           "SELECT short_code FROM urls WHERE expires_at < NOW()"
       )
       # Batch delete from database
       # Invalidate cache entries
   ```

3. **TTL in Cache**:
   - Set Redis TTL based on URL expiration
   - Automatic cleanup by Redis
   - No manual cache invalidation needed

4. **Partitioned Cleanup**:
   - Partition by creation_date or expiry_date
   - Drop entire partitions when expired
   - Faster than row-by-row deletion"

### Q5: "How do you prevent abuse and malicious URLs?"
**A**: "Multi-layer security approach:

1. **Input Validation**:
   ```python
   def validate_url(url):
       # Check format with regex
       if not url_pattern.match(url):
           return False
       
       # Check against domain blacklist
       domain = extract_domain(url)
       if domain in blocked_domains:
           return False
       
       # Check for malicious patterns
       malicious_patterns = ['javascript:', 'data:', 'vbscript:']
       return not any(pattern in url.lower() for pattern in malicious_patterns)
   ```

2. **Rate Limiting**:
   - Per-IP limits: 100 URLs/hour for anonymous users
   - Per-user limits: 1000 URLs/hour for authenticated users
   - Sliding window rate limiter with Redis

3. **Real-time URL Scanning**:
   ```python
   async def scan_url_security(url):
       # Check against Google Safe Browsing API
       # Scan with VirusTotal API  
       # Internal machine learning model
       if is_malicious(url):
           mark_url_as_malicious(short_code)
   ```

4. **Behavioral Analysis**:
   - Detect patterns: too many URLs from same IP
   - Monitor click patterns for suspicious activity
   - Automatic temporary bans for abuse

5. **User Reporting**: Allow users to report malicious URLs
6. **Content Monitoring**: Periodic re-scanning of stored URLs"

### Q6: "How do you design for high availability (99.9% uptime)?"
**A**: "Fault-tolerant architecture with redundancy:

1. **Multi-Region Deployment**:
   - Active-active setup across 3+ AWS regions
   - Route53 health checks for automatic failover
   - Cross-region database replication

2. **Database High Availability**:
   ```
   Primary Region:     Secondary Region:
   Master DB           Read Replica → Master (failover)
   Read Replicas (3)   Read Replicas (2)
   ```

3. **Circuit Breakers**:
   ```python
   @circuit_breaker(failure_threshold=5, timeout=30)
   def get_from_database(short_code):
       # If DB fails 5 times, circuit opens
       # Serve from cache only for 30 seconds
   ```

4. **Graceful Degradation**:
   - Redirect works even if analytics fails
   - Serve from cache if database is down
   - Disable URL creation if database unavailable

5. **Health Monitoring**:
   - Application health endpoints
   - Database connection monitoring
   - Cache availability checks
   - Automated alerting on failures

6. **Data Backup & Recovery**:
   - Continuous database backups
   - Point-in-time recovery capability
   - Regular disaster recovery testing"

### Q7: "How do you handle custom aliases and prevent conflicts?"
**A**: "Namespace management with reservation system:

1. **Alias Validation**:
   ```python
   def validate_custom_alias(alias):
       # Length constraints
       if len(alias) < 3 or len(alias) > 20:
           return False
       
       # Character restrictions  
       if not re.match(r'^[a-zA-Z0-9_-]+$', alias):
           return False
       
       # Reserved words
       reserved = ['api', 'www', 'admin', 'help']
       if alias.lower() in reserved:
           return False
       
       return True
   ```

2. **Atomic Reservation**:
   ```sql
   BEGIN TRANSACTION;
   
   -- Check availability
   SELECT short_code FROM urls WHERE short_code = 'custom-alias' FOR UPDATE;
   
   -- If not exists, insert
   INSERT INTO urls (short_code, long_url, custom_alias) 
   VALUES ('custom-alias', 'https://example.com', true);
   
   COMMIT;
   ```

3. **Premium Aliases**:
   - Short aliases (3-4 chars) for premium users
   - Dictionary words reserved for enterprise
   - Pricing tiers based on alias desirability

4. **Namespace Segregation**:
   - User prefixes: 'user123-mylink'
   - Organization namespaces: 'company/product-link'
   - Prevents conflicts between users"

## 🧪 Testing Strategy

### Unit Tests
```python
def test_base62_encoding():
    encoder = Base62Encoder()
    
    # Test basic encoding/decoding
    assert encoder.decode(encoder.encode(12345)) == 12345
    
    # Test edge cases
    assert encoder.encode(0) == "0"
    assert len(encoder.encode(999999)) == 6  # Fixed length

def test_url_validation():
    validator = URLValidator()
    
    # Valid URLs
    assert validator.is_valid_url("https://www.google.com")
    assert validator.is_valid_url("http://localhost:3000/api")
    
    # Invalid URLs
    assert not validator.is_valid_url("javascript:alert(1)")
    assert not validator.is_valid_url("not-a-url")
    assert not validator.is_valid_url("")

def test_collision_handling():
    service = URLShortenerService(config)
    
    # Mock encoder to return same code twice
    with mock.patch.object(service.encoder, 'generate_random_code') as mock_gen:
        mock_gen.side_effect = ['abc123', 'def456']  # First collides, second succeeds
        
        # First URL gets abc123
        response1 = service.shorten_url("https://example.com")
        assert response1.short_code == 'abc123'
        
        # Second URL should get def456 (collision handled)
        response2 = service.shorten_url("https://google.com")
        assert response2.short_code == 'def456'
```

### Integration Tests
```python
def test_end_to_end_flow():
    service = URLShortenerService(config)
    
    # Shorten URL
    shorten_response = service.shorten_url(
        "https://www.example.com/very-long-url?param=value",
        user_id="test_user"
    )
    assert shorten_response.success
    
    # Redirect URL  
    redirect_response = service.redirect_url(shorten_response.short_code)
    assert redirect_response.success
    assert redirect_response.long_url == "https://www.example.com/very-long-url?param=value"
    
    # Check analytics
    analytics = service.get_analytics(shorten_response.short_code)
    assert analytics.total_clicks == 1

def test_database_integration():
    # Test with real database connection
    # Verify data persistence
    # Test transaction rollbacks
    pass

def test_cache_integration():
    # Test Redis cache behavior
    # Verify cache hit/miss scenarios
    # Test cache invalidation
    pass
```

### Load Tests
```python
import asyncio
import aiohttp

async def load_test_redirects():
    """Test redirect performance under load"""
    
    # Create test URLs
    test_codes = setup_test_urls(count=1000)
    
    async def redirect_request(session, short_code):
        async with session.get(f'/r/{short_code}') as response:
            return response.status
    
    # Concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(10000):  # 10K concurrent requests
            code = random.choice(test_codes)
            tasks.append(redirect_request(session, code))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
    
    # Analyze results
    success_rate = sum(1 for status in results if status == 302) / len(results)
    rps = len(results) / duration
    
    assert success_rate > 0.99  # 99% success rate
    assert rps > 1000  # 1000+ requests per second
```

## 🚀 Production Considerations

### Database Selection & Schema

#### PostgreSQL (Recommended for consistency)
```sql
-- Optimized schema for PostgreSQL
CREATE TABLE urls (
    short_code VARCHAR(10) PRIMARY KEY,
    long_url TEXT NOT NULL CHECK (length(long_url) <= 2048),
    user_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    click_count BIGINT DEFAULT 0,
    status SMALLINT DEFAULT 1, -- 1=active, 2=expired, 3=disabled
    custom_alias BOOLEAN DEFAULT FALSE,
    
    -- Indexes for common queries
    CONSTRAINT valid_short_code CHECK (short_code ~ '^[a-zA-Z0-9_-]{3,10}$')
);

CREATE INDEX CONCURRENTLY idx_user_created ON urls (user_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_expires_cleanup ON urls (expires_at) WHERE expires_at IS NOT NULL;

-- Partitioning by creation date for easier maintenance
CREATE TABLE urls_y2024m01 PARTITION OF urls 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

#### Cassandra (For massive scale)
```sql
-- Optimized for high write throughput
CREATE TABLE urls (
    short_code text PRIMARY KEY,
    long_url text,
    user_id text,
    created_at timestamp,
    expires_at timestamp,
    click_count counter
);

-- Separate table for user queries
CREATE TABLE user_urls (
    user_id text,
    created_at timestamp,
    short_code text,
    PRIMARY KEY (user_id, created_at, short_code)
) WITH CLUSTERING ORDER BY (created_at DESC);
```

### Caching Architecture

#### Redis Configuration
```redis
# redis.conf optimizations
maxmemory 8gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Cluster setup for high availability
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
```

#### Cache Key Strategy
```python
CACHE_PATTERNS = {
    'url_mapping': 'url:{short_code}',        # TTL: 1 hour
    'user_urls': 'user:{user_id}:urls',       # TTL: 15 minutes  
    'analytics': 'analytics:{short_code}',     # TTL: 5 minutes
    'rate_limit': 'rate:{user_id}:{window}',  # TTL: window size
}

def cache_key(pattern: str, **kwargs) -> str:
    return CACHE_PATTERNS[pattern].format(**kwargs)
```

### Monitoring & Observability

#### Key Metrics to Track
```python
# Application metrics
METRICS = {
    # Business metrics
    'urls_created_per_minute': Counter,
    'redirects_per_second': Counter,
    'cache_hit_rate': Gauge,
    
    # Performance metrics  
    'redirect_latency_p95': Histogram,
    'database_query_time': Histogram,
    'cache_operation_time': Histogram,
    
    # Error metrics
    'invalid_url_requests': Counter,
    'database_errors': Counter,
    'cache_errors': Counter,
    
    # Resource metrics
    'active_database_connections': Gauge,
    'memory_usage_percent': Gauge,
    'cpu_usage_percent': Gauge
}
```

#### Alerting Rules
```yaml
# Example Prometheus alerts
groups:
  - name: url-shortener
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "High error rate detected"
          
      - alert: DatabaseConnectionPoolExhausted
        expr: database_connections_active / database_connections_max > 0.9
        for: 1m
        annotations:
          summary: "Database connection pool nearly exhausted"
          
      - alert: CacheHitRateLow
        expr: cache_hit_rate < 0.8
        for: 5m
        annotations:
          summary: "Cache hit rate below 80%"
```

### Security Implementation

#### Input Sanitization
```python
class SecurityValidator:
    def __init__(self):
        # Compile regexes for performance
        self.url_pattern = re.compile(
            r'^https?://(?:[a-zA-Z0-9-_.]+\.)+[a-zA-Z]{2,}(?:/.*)?$'
        )
        self.malicious_patterns = [
            re.compile(r'javascript:', re.I),
            re.compile(r'data:', re.I),
            re.compile(r'vbscript:', re.I),
        ]
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        # Length check
        if len(url) > 2048:
            return False, "URL too long"
        
        # Format check
        if not self.url_pattern.match(url):
            return False, "Invalid URL format"
        
        # Malicious pattern check
        for pattern in self.malicious_patterns:
            if pattern.search(url):
                return False, "Potentially malicious URL"
        
        return True, "Valid"
```

#### Rate Limiting Implementation
```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def is_allowed(self, user_id: str, limit: int, window: int) -> bool:
        """Sliding window rate limiter"""
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(f"rate:{user_id}", 0, now - window)
        
        # Count current requests
        pipeline.zcard(f"rate:{user_id}")
        
        # Add current request
        pipeline.zadd(f"rate:{user_id}", {str(now): now})
        
        # Set expiry
        pipeline.expire(f"rate:{user_id}", window)
        
        results = await pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit
```

---

## 💡 Final Interview Tips

1. **Start with clarifying questions**: Scale, features, constraints matter
2. **Design incrementally**: Basic version → Advanced features  
3. **Discuss trade-offs**: Every decision has alternatives
4. **Consider failure modes**: What breaks and how to handle it
5. **Think about operations**: Monitoring, debugging, scaling
6. **Know the numbers**: 
   - Base62^6 = 56.8 billion combinations
   - Typical read:write ratio = 100:1
   - Target latency: <100ms for redirects

**Most Important**: Show systematic thinking about real-world production systems. URL shorteners seem simple but have deep scalability and reliability challenges that make them excellent interview problems.

## 📚 Additional Resources

### Similar Systems to Study
- **bit.ly**: Analytics dashboard, custom domains
- **t.co (Twitter)**: High-scale real-time processing
- **goo.gl (Google)**: QR code integration, click fraud detection
- **tinyurl.com**: Simple interface, long history

### Technologies to Explore
- **Databases**: PostgreSQL sharding, Cassandra modeling
- **Caching**: Redis Cluster, Memcached
- **Analytics**: Apache Kafka, ClickHouse, ElasticSearch  
- **Monitoring**: Prometheus, Grafana, Jaeger tracing