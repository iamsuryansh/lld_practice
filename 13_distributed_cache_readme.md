# Distributed Cache System - Low Level Design

## Problem Statement

Design and implement a **Distributed Cache System** that supports:

1. **Horizontal scalability** with multiple cache nodes
2. **Data partitioning** using consistent hashing
3. **Replication** for fault tolerance (configurable replication factor)
4. **Configurable consistency** levels (ONE, QUORUM, ALL)
5. **Automatic failover** when nodes crash
6. **Read repair** for eventual consistency
7. **Hot key detection** for skewed workloads
8. **TTL-based expiration** with lazy eviction

### Core Requirements

- Support 1000s of nodes in a cluster
- Sub-millisecond latency for cache operations
- Graceful handling of node failures (no data loss with replication)
- Minimal data movement on cluster topology changes
- CAP theorem trade-offs (choose between consistency and availability)
- Hot key mitigation strategies

### Technical Constraints

- Python 3.8+ (standard library only)
- In-memory storage (no external databases)
- Thread-safe operations
- O(log N) node lookup using consistent hashing
- Quorum-based reads/writes for consistency

---

## Step-by-Step Implementation Guide

### Phase 1: Core Models and Data Structures (20 minutes)

**What to build:**
- Enums for consistency levels and replication strategies
- Data classes for CacheEntry, CacheNode, ReadResult, WriteResult
- Foundation for distributed coordination

**Interview Focus:**
- Why version numbers in CacheEntry? (Conflict resolution, optimistic locking)
- What metadata is essential? (Timestamp for TTL, version for consistency)
- How to model node state? (Health status, capacity, location)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time

class ConsistencyLevel(Enum):
    """CAP theorem trade-offs"""
    ONE = "one"        # Fast, may be stale
    QUORUM = "quorum"  # Balanced (industry standard)
    ALL = "all"        # Slow, most consistent

@dataclass
class CacheEntry:
    """Value with versioning for conflict resolution"""
    key: str
    value: Any
    version: int = 0                    # Optimistic locking
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds

@dataclass
class CacheNode:
    """Physical cache server"""
    node_id: str
    host: str
    port: int
    is_alive: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    memory_used: int = 0
    memory_limit: int = 1_000_000_000  # 1GB
```

**Key Design Decisions:**
- **Version numbers**: Enable last-write-wins conflict resolution
- **TTL per entry**: Flexible expiration policies
- **Node health tracking**: Automatic failure detection via heartbeats
- **Memory limits**: Prevent OOM, trigger eviction

---

### Phase 2: Consistent Hashing Ring (30 minutes)

**What to build:**
- Hash ring with virtual nodes for load balancing
- Node addition/removal with minimal remapping
- Replica placement algorithm

**Interview Focus:**
- Why consistent hashing vs modulo hashing? (Minimal remapping: 1/N keys vs all keys)
- What are virtual nodes? (Better distribution, handles heterogeneous node capacities)
- How to ensure replica diversity? (Skip virtual nodes of same physical node)

```python
import hashlib
import bisect
from typing import Dict, List, Set

class ConsistentHashRing:
    """
    Consistent hashing for data partitioning
    
    Interview Focus: Compare with simple hash % N
    - Modulo: Adding/removing node remaps ALL keys
    - Consistent: Only 1/N keys remapped (neighbors)
    """
    
    def __init__(self, virtual_nodes_per_node: int = 150):
        """
        Args:
            virtual_nodes_per_node: More virtual nodes = better distribution
                                   Typical: 100-200 per physical node
        """
        self.virtual_nodes = virtual_nodes_per_node
        self._ring: Dict[int, CacheNode] = {}    # hash → node
        self._sorted_hashes: List[int] = []      # Binary search
        self._nodes: Set[CacheNode] = set()
        self._lock = threading.RLock()
    
    def _hash(self, key: str) -> int:
        """MD5 for uniform distribution (128 bits)"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: CacheNode) -> int:
        """Add node with virtual replicas"""
        with self._lock:
            self._nodes.add(node)
            
            # Add virtual nodes: node_id:0, node_id:1, ...
            for i in range(self.virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = self._hash(virtual_key)
                
                self._ring[hash_value] = node
                bisect.insort(self._sorted_hashes, hash_value)
            
            # Return keys needing remapping (for data migration)
            return self.virtual_nodes
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        """
        Find node for key (clockwise search in ring)
        
        Interview Q: Why clockwise?
        A: Convention - could be counter-clockwise, just be consistent
        """
        with self._lock:
            if not self._sorted_hashes:
                return None
            
            hash_value = self._hash(key)
            
            # Binary search: first position >= hash_value
            idx = bisect.bisect_right(self._sorted_hashes, hash_value)
            
            # Wrap around to start
            if idx == len(self._sorted_hashes):
                idx = 0
            
            return self._ring[self._sorted_hashes[idx]]
    
    def get_replica_nodes(self, key: str, replication_factor: int) -> List[CacheNode]:
        """
        Get N unique physical nodes for replication
        
        Interview Focus: Walk clockwise, skip duplicate physical nodes
        """
        with self._lock:
            hash_value = self._hash(key)
            idx = bisect.bisect_right(self._sorted_hashes, hash_value)
            
            unique_nodes: List[CacheNode] = []
            seen_ids: Set[str] = set()
            
            while len(unique_nodes) < replication_factor:
                if idx >= len(self._sorted_hashes):
                    idx = 0
                
                node = self._ring[self._sorted_hashes[idx]]
                
                if node.node_id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.node_id)
                
                idx += 1
            
            return unique_nodes
```

**Complexity Analysis:**
- `add_node`: O(V log N) where V = virtual nodes, N = total positions
- `get_node`: O(log N) binary search
- `get_replica_nodes`: O(R * log N) where R = replication factor

**Why better than modulo?**
```python
# Modulo hashing: hash(key) % num_nodes
# Problem: Adding node 4 to 3-node cluster remaps 75% of keys!
# key → node_id
# "user:1" → 0  becomes  "user:1" → 1  (remapped)
# "user:2" → 1  becomes  "user:2" → 2  (remapped)
# "user:3" → 2  becomes  "user:3" → 3  (remapped)

# Consistent hashing: Only ~25% remapped (neighbors only)
```

---

### Phase 3: Local Cache Storage (25 minutes)

**What to build:**
- In-memory storage with LRU eviction
- TTL-based lazy expiration
- Thread-safe operations with statistics

**Interview Focus:**
- Why LRU for cache? (Temporal locality, simple, predictable)
- When to check TTL? (Lazy on read vs eager background cleanup)
- Trade-off: Memory vs CPU (more memory = fewer evictions)

```python
from typing import Dict, List, Optional
import threading

class LocalCacheStorage:
    """Single-node cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._storage: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU queue
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get with LRU update and TTL check"""
        with self._lock:
            if key not in self._storage:
                self._stats["misses"] += 1
                return None
            
            entry = self._storage[key]
            
            # Lazy TTL check
            if entry.is_expired():
                del self._storage[key]
                self._access_order.remove(key)
                self._stats["misses"] += 1
                return None
            
            # Update LRU (move to end)
            self._access_order.remove(key)
            self._access_order.append(key)
            self._stats["hits"] += 1
            
            return entry
    
    def put(self, entry: CacheEntry) -> bool:
        """Put with eviction if at capacity"""
        with self._lock:
            # Evict LRU if needed
            if entry.key not in self._storage and len(self._storage) >= self.max_size:
                self._evict_lru()
            
            # Update or insert
            if entry.key in self._storage:
                self._access_order.remove(entry.key)
            
            self._storage[entry.key] = entry
            self._access_order.append(entry.key)
            
            return True
    
    def _evict_lru(self) -> None:
        """Remove least recently used entry"""
        if self._access_order:
            lru_key = self._access_order[0]
            del self._storage[lru_key]
            self._access_order.pop(0)
            self._stats["evictions"] += 1
```

**Alternative: LFU (Least Frequently Used)**
```python
# Use heap to track frequency
import heapq

class LFUCache:
    def __init__(self):
        self._frequency: Dict[str, int] = {}
        self._heap: List[Tuple[int, str]] = []  # (freq, key)
    
    def _evict_lfu(self):
        while self._heap:
            freq, key = heapq.heappop(self._heap)
            if key in self._storage and self._frequency[key] == freq:
                del self._storage[key]
                del self._frequency[key]
                break
```

**LRU vs LFU Trade-offs:**
| Policy | Use Case | Complexity | Memory |
|--------|----------|------------|--------|
| LRU | Recent access matters | O(1) get/put | Low |
| LFU | Frequency matters | O(log N) evict | Higher |
| TTL | Time-based expiration | O(1) lazy check | Low |

---

### Phase 4: Distributed Coordination (40 minutes)

**What to build:**
- Quorum-based reads/writes
- Read repair mechanism
- Conflict resolution using versions

**Interview Focus:**
- What is quorum? (Majority: N/2 + 1, ensures consistency)
- Why read repair? (Fix stale replicas detected during reads)
- How to resolve conflicts? (Last-write-wins using version numbers)

```python
class DistributedCache:
    """Distributed cache with replication and consistency"""
    
    def __init__(
        self,
        cluster_name: str,
        replication_factor: int = 3,
        default_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    ):
        self.replication_factor = replication_factor
        self.default_consistency = default_consistency
        self.ring = ConsistentHashRing()
        self._node_storage: Dict[str, LocalCacheStorage] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, consistency: Optional[ConsistencyLevel] = None) -> ReadResult:
        """
        Distributed read with quorum
        
        Interview Focus: Explain algorithm
        1. Find replicas using consistent hashing
        2. Read from N replicas (N depends on consistency level)
        3. Return latest version (highest version number)
        4. Asynchronously repair stale replicas
        """
        with self._lock:
            consistency = consistency or self.default_consistency
            
            # Find replica nodes
            replicas = self.ring.get_replica_nodes(key, self.replication_factor)
            
            if not replicas:
                return ReadResult(success=False, message="No nodes available")
            
            # Determine how many to read
            required_reads = self._get_required_responses(consistency, len(replicas))
            
            # Read from replicas
            entries: List[Tuple[CacheEntry, str]] = []
            for node in replicas[:required_reads]:
                storage = self._node_storage.get(node.node_id)
                if storage:
                    entry = storage.get(key)
                    if entry:
                        entries.append((entry, node.node_id))
            
            if not entries:
                return ReadResult(success=False, message="Key not found")
            
            # Find latest version
            latest_entry = max(entries, key=lambda x: x[0].version)[0]
            
            # Read repair (fix stale replicas)
            self._read_repair(key, latest_entry, entries, replicas)
            
            return ReadResult(
                success=True,
                value=latest_entry.value,
                version=latest_entry.version,
                replicas_read=len(entries)
            )
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None,
            consistency: Optional[ConsistencyLevel] = None) -> WriteResult:
        """
        Distributed write with quorum
        
        Interview Focus: Version increment for conflict resolution
        """
        with self._lock:
            consistency = consistency or self.default_consistency
            
            # Find replicas
            replicas = self.ring.get_replica_nodes(key, self.replication_factor)
            
            # Get current version (optimistic locking)
            current_version = 0
            for node in replicas:
                storage = self._node_storage.get(node.node_id)
                if storage:
                    entry = storage.get(key)
                    if entry:
                        current_version = max(current_version, entry.version)
            
            # Create new entry with incremented version
            new_entry = CacheEntry(
                key=key,
                value=value,
                version=current_version + 1,
                ttl_seconds=ttl_seconds
            )
            
            # Write to replicas
            successful_writes = 0
            for node in replicas:
                storage = self._node_storage.get(node.node_id)
                if storage and storage.put(new_entry):
                    successful_writes += 1
            
            # Check quorum
            required_writes = self._get_required_responses(consistency, len(replicas))
            success = successful_writes >= required_writes
            
            return WriteResult(
                success=success,
                replicas_written=successful_writes,
                message="OK" if success else f"Quorum not met: {successful_writes}/{required_writes}"
            )
    
    def _get_required_responses(self, consistency: ConsistencyLevel, total_replicas: int) -> int:
        """
        Quorum calculation
        
        Interview Q: Why N/2 + 1 for QUORUM?
        A: Ensures overlap between read/write quorums, guaranteeing consistency
        
        Example: 5 replicas
        - Write quorum = 3
        - Read quorum = 3
        - Overlap = at least 1 node has latest write
        """
        if consistency == ConsistencyLevel.ONE:
            return 1
        elif consistency == ConsistencyLevel.QUORUM:
            return (total_replicas // 2) + 1
        elif consistency == ConsistencyLevel.ALL:
            return total_replicas
        return 1
    
    def _read_repair(self, key: str, latest_entry: CacheEntry,
                     entries: List[Tuple[CacheEntry, str]],
                     replicas: List[CacheNode]) -> None:
        """
        Read repair: Fix stale replicas in background
        
        Interview Focus: Eventual consistency mechanism
        - Detected during reads (no extra RPC)
        - Asynchronous (no user latency impact)
        - Convergence guarantee
        """
        # Find stale replicas
        stale_versions = [e for e, _ in entries if e.version < latest_entry.version]
        
        if not stale_versions:
            return
        
        # Update stale replicas
        for node in replicas:
            storage = self._node_storage.get(node.node_id)
            if storage:
                current = storage.get(key)
                if not current or current.version < latest_entry.version:
                    storage.put(latest_entry)
```

**Quorum Math:**
```
Replication factor R = 3
Write quorum W = 2
Read quorum Rd = 2

Requirement: W + Rd > R
2 + 2 > 3 ✓

This ensures reads always see latest write:
- Write succeeds on nodes {A, B}
- Read from {B, C} will include B (has latest)
```

---

### Phase 5: Hot Key Detection (20 minutes)

**What to build:**
- Sliding window access tracking
- Threshold-based hot key identification
- Mitigation strategies

**Interview Focus:**
- Why are hot keys problematic? (Single node overload, uneven load)
- Solutions: Local caching, key splitting, increased replication
- How to detect? (Count accesses in time window)

```python
from collections import defaultdict
from typing import Dict, List, Tuple
import time

class HotKeyDetector:
    """
    Detect frequently accessed keys
    
    Interview Focus: Hot key problems
    - Celebrity problem: One user followed by millions
    - Trending content: Viral video/tweet
    - Impact: Single node handles all traffic → bottleneck
    """
    
    def __init__(self, window_seconds: int = 60, threshold: int = 1000):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self._access_counts: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_access(self, key: str) -> None:
        """Record key access with timestamp"""
        with self._lock:
            now = time.time()
            self._access_counts[key].append(now)
            
            # Cleanup old timestamps (sliding window)
            cutoff = now - self.window_seconds
            self._access_counts[key] = [
                ts for ts in self._access_counts[key] if ts >= cutoff
            ]
    
    def get_hot_keys(self) -> List[Tuple[str, int]]:
        """Get keys exceeding threshold"""
        with self._lock:
            hot_keys = []
            for key, timestamps in self._access_counts.items():
                count = len(timestamps)
                if count >= self.threshold:
                    hot_keys.append((key, count))
            
            return sorted(hot_keys, key=lambda x: x[1], reverse=True)
    
    def is_hot_key(self, key: str) -> bool:
        """Check if key is hot"""
        with self._lock:
            return len(self._access_counts.get(key, [])) >= self.threshold
```

**Hot Key Mitigation Strategies:**

1. **Local Caching**: Cache hot keys on application tier
```python
class LocalHotKeyCache:
    def __init__(self, ttl_seconds: int = 10):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
        return None
```

2. **Key Splitting**: Replicate hot key with random suffix
```python
def get_hot_key_with_splitting(key: str) -> Any:
    if is_hot_key(key):
        # Read from random replica
        replica_key = f"{key}:replica:{random.randint(0, 9)}"
        return cache.get(replica_key)
    return cache.get(key)
```

3. **Increased Replication**: More replicas for hot keys
```python
if is_hot_key(key):
    replication_factor = 10  # Instead of default 3
```

---

### Phase 6: Node Failure Handling (25 minutes)

**What to build:**
- Heartbeat-based failure detection
- Automatic failover to replica nodes
- Data rebalancing on topology changes

**Interview Focus:**
- How to detect failures? (Heartbeat timeout, gossip protocol)
- What happens to data on failed node? (Read from replicas)
- How to add new nodes? (Consistent hashing minimizes data movement)

```python
import threading
import time

class HealthMonitor:
    """Monitor node health with heartbeats"""
    
    def __init__(self, heartbeat_interval: int = 5, failure_threshold: int = 3):
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold  # Missed heartbeats
        self._nodes: Dict[str, CacheNode] = {}
        self._lock = threading.Lock()
        self._running = False
    
    def start_monitoring(self):
        """Start background heartbeat checker"""
        self._running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """Check node health periodically"""
        while self._running:
            time.sleep(self.heartbeat_interval)
            self._check_nodes()
    
    def _check_nodes(self):
        """Mark nodes as dead if heartbeat missed"""
        with self._lock:
            now = time.time()
            for node in self._nodes.values():
                time_since_heartbeat = now - node.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_interval * self.failure_threshold:
                    if node.is_alive:
                        node.is_alive = False
                        print(f"Node {node.node_id} marked as DEAD")
                        # Trigger rebalancing
                        self._handle_node_failure(node)
    
    def _handle_node_failure(self, node: CacheNode):
        """Remove failed node from ring"""
        # In production: Trigger data migration from replicas
        pass
```

**Failure Scenarios:**

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Node crash | Heartbeat timeout | Read from replicas |
| Network partition | Gossip protocol | Split-brain handling |
| Slow node | Latency monitoring | Route around slow node |
| Data corruption | Checksum validation | Restore from replica |

---

## Critical Knowledge Points

### 1. CAP Theorem Trade-offs

**Theorem**: In a distributed system, you can only guarantee 2 of 3:
- **C**onsistency: All nodes see same data
- **A**vailability: Every request gets response
- **P**artition tolerance: System works despite network failures

**Our Implementation:**
- **CP mode** (QUORUM, ALL): Consistent but may reject writes during partition
- **AP mode** (ONE): Available but may serve stale data

**Interview Q: How to choose?**
```
Financial transactions → CP (consistency critical)
Social media feeds → AP (availability critical)
Shopping cart → AP (availability preferred)
```

### 2. Consistent Hashing Benefits

**Problem with Modulo Hashing:**
```python
# 3 nodes: hash(key) % 3
# Add 4th node: hash(key) % 4
# Result: 75% of keys remapped!
```

**Consistent Hashing:**
- Only **K/N** keys remapped (K = total keys, N = nodes)
- Average: **1/N** of keys move per node change
- Example: Adding 4th node to 3-node cluster remaps ~25% (only neighbors)

### 3. Quorum Math

**Formula**: W + R > N (W = write quorum, R = read quorum, N = replicas)

**Examples:**
```
N=3, W=2, R=2: 2+2>3 ✓ (balanced)
N=5, W=3, R=3: 3+3>5 ✓ (standard)
N=5, W=1, R=5: 1+5>5 ✓ (fast writes, slow reads)
N=5, W=5, R=1: 5+1>5 ✓ (slow writes, fast reads)
```

### 4. Read Repair vs Anti-Entropy

**Read Repair** (our implementation):
- Triggered during reads
- Low overhead
- Eventual consistency
- May miss keys never read

**Anti-Entropy** (Merkle trees):
- Background process
- Compares all data
- Guaranteed consistency
- Higher overhead

### 5. Hot Key Impact

**Example:**
```
Normal key: 100 req/sec → 1 node handles easily
Hot key: 100,000 req/sec → 1 node overloaded

Solutions:
1. Local cache: 99% hit rate → 1,000 req/sec to backend
2. Key splitting: 10 replicas → 10,000 req/sec per node
3. Increased replication: 20 replicas → 5,000 req/sec per node
```

---

## Expected Interview Questions

### Q1: Why use consistent hashing instead of simple modulo hashing?

**Answer:**

**Modulo hashing problem:**
```python
# 3-node cluster
node = hash(key) % 3

# Add 4th node
node = hash(key) % 4
# Result: 75% of keys remapped!
```

**Consistent hashing benefits:**
1. **Minimal remapping**: Only K/N keys move (K = keys, N = nodes)
2. **Even distribution**: Virtual nodes balance load
3. **Incremental scalability**: Add nodes without rehashing all keys

**Example:**
```
100,000 keys on 3 nodes
Add 4th node:
- Modulo: 75,000 keys remapped
- Consistent: ~25,000 keys remapped (only from neighbors)
```

**Virtual nodes:**
```python
# Without virtual nodes: Uneven distribution
node1: 40,000 keys
node2: 35,000 keys
node3: 25,000 keys

# With 150 virtual nodes per physical node: Even distribution
node1: 33,333 keys
node2: 33,333 keys
node3: 33,334 keys
```

---

### Q2: How does quorum-based consistency work? What's the math behind it?

**Answer:**

**Quorum requirement:** W + R > N

Where:
- W = Write quorum (nodes that must acknowledge write)
- R = Read quorum (nodes to read from)
- N = Replication factor

**Why this works:**
```
N = 3 replicas {A, B, C}
W = 2 (write to 2 nodes)
R = 2 (read from 2 nodes)

Write scenario:
- Client writes to {A, B} (W=2 satisfied)
- Node C is stale

Read scenario:
- Client reads from {B, C}
- At least one (B) has latest write
- W + R = 4 > N = 3 guarantees overlap
```

**Trade-offs:**

| Configuration | Use Case | Latency | Consistency |
|---------------|----------|---------|-------------|
| W=1, R=N | Write-heavy | Low write | Strong read |
| W=N, R=1 | Read-heavy | Low read | Strong write |
| W=Q, R=Q | Balanced | Medium | Strong both |

**Example configurations for N=5:**
```python
# Fast writes, slower reads
W = 1, R = 5  # 1+5 > 5 ✓

# Fast reads, slower writes
W = 5, R = 1  # 5+1 > 5 ✓

# Balanced (standard)
W = 3, R = 3  # 3+3 > 5 ✓

# Invalid (no overlap guarantee)
W = 2, R = 2  # 2+2 = 4 < 5 ✗
```

---

### Q3: Explain read repair. When does it happen and why is it needed?

**Answer:**

**Read repair** fixes stale replicas detected during read operations.

**Algorithm:**
```python
def get_with_read_repair(key):
    # 1. Read from quorum (e.g., 2 of 3 replicas)
    replica1_value = read_from_node1(key)  # version 5
    replica2_value = read_from_node2(key)  # version 3 (stale!)
    
    # 2. Find latest version
    latest = max(replica1_value, replica2_value, key=lambda x: x.version)
    # latest.version = 5
    
    # 3. Return latest to client
    return latest.value
    
    # 4. Asynchronously repair stale replicas
    if replica2_value.version < latest.version:
        async_write_to_node2(latest)  # Update node2 to version 5
```

**When it happens:**
- During **reads** (piggybacked on read operation)
- When versions mismatch detected
- No additional RPC needed

**Why needed:**
- **Eventual consistency**: Writes may not reach all replicas
- **Network partitions**: Some nodes temporarily unreachable
- **Faster writes**: ONE consistency allows incomplete replication

**Example scenario:**
```
Time 0: Write "user:123" to nodes {A, B} with W=2 (node C missed)
  A: version 5
  B: version 5
  C: version 4 (stale)

Time 1: Read "user:123" with R=2 from nodes {B, C}
  B returns version 5
  C returns version 4
  → Read repair updates C to version 5
```

**Alternative: Anti-entropy (Merkle trees)**
- Background process
- Compares all data periodically
- Higher overhead but guaranteed convergence

---

### Q4: How would you handle hot keys (celebrity problem)?

**Answer:**

**Problem:**
```
Normal key: 100 req/sec → 1 node handles easily
Hot key: 100,000 req/sec → 1 node bottleneck

Example: Celebrity user followed by 10M users
→ All reads hit same node
```

**Detection:**
```python
class HotKeyDetector:
    def __init__(self, threshold: int = 1000):  # 1000 req/min
        self._access_counts: Dict[str, List[float]] = {}
    
    def record_access(self, key: str):
        now = time.time()
        self._access_counts[key].append(now)
        
        # Keep last 60 seconds
        self._access_counts[key] = [
            ts for ts in self._access_counts[key]
            if now - ts < 60
        ]
    
    def is_hot(self, key: str) -> bool:
        return len(self._access_counts[key]) > self.threshold
```

**Mitigation strategies:**

**1. Local Caching (Application-level)**
```python
local_cache = LRU(capacity=1000, ttl=10)  # 10-second TTL

def get_with_local_cache(key):
    # Check local first
    value = local_cache.get(key)
    if value:
        return value
    
    # Fetch from distributed cache
    value = distributed_cache.get(key)
    
    # Cache hot keys locally
    if hot_key_detector.is_hot(key):
        local_cache.put(key, value)
    
    return value

# Result: 99% hit rate locally → 100k → 1k backend requests
```

**2. Key Splitting (Multiple Replicas)**
```python
def get_hot_key_with_splitting(key: str):
    if hot_key_detector.is_hot(key):
        # Replicate key with 10 suffixes
        replica_id = random.randint(0, 9)
        replica_key = f"{key}:replica:{replica_id}"
        return cache.get(replica_key)
    
    return cache.get(key)

# On write: Update all replicas
def put_hot_key(key: str, value: Any):
    if hot_key_detector.is_hot(key):
        for i in range(10):
            cache.put(f"{key}:replica:{i}", value)
    else:
        cache.put(key, value)

# Result: 100k req/sec → 10k req/sec per replica node
```

**3. Increased Replication Factor**
```python
def get_replication_factor(key: str) -> int:
    if hot_key_detector.is_hot(key):
        return 20  # More replicas for hot keys
    return 3  # Default

# Distribute load across 20 nodes instead of 3
```

**4. CDN/Edge Caching**
```
Hot static content → CDN (Cloudflare, Akamai)
- Serve from edge locations
- Reduce backend load to zero
```

**Comparison:**

| Solution | Complexity | Effectiveness | Use Case |
|----------|------------|---------------|----------|
| Local cache | Low | High (99% reduction) | User sessions |
| Key splitting | Medium | High | Read-heavy |
| Increased replication | Low | Medium | Dynamic content |
| CDN | Low | Very high | Static content |

---

### Q5: What happens when a node fails? How do you ensure no data loss?

**Answer:**

**Failure Detection:**
```python
class HealthMonitor:
    def __init__(self, heartbeat_interval=5, failure_threshold=3):
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
    
    def monitor_node(self, node: CacheNode):
        missed_heartbeats = 0
        
        while True:
            if not receive_heartbeat(node, timeout=self.heartbeat_interval):
                missed_heartbeats += 1
                
                if missed_heartbeats >= self.failure_threshold:
                    # 3 consecutive failures → mark as dead
                    mark_node_dead(node)
                    trigger_failover(node)
                    break
            else:
                missed_heartbeats = 0
```

**Failover Process:**

**1. Immediate Failover (no data loss due to replication)**
```python
# Replication factor = 3
# Key "user:123" stored on nodes {A, B, C}

# Node A fails
# Consistent hashing: Find new primary
new_primary = ring.get_node("user:123")  # Returns B

# Reads still work (from replicas B, C)
value = cache.get("user:123")  # Reads from B or C
```

**2. Remove Failed Node from Ring**
```python
def handle_node_failure(failed_node: CacheNode):
    # Remove from consistent hash ring
    ring.remove_node(failed_node)
    
    # Keys affected: Only those where failed_node was replica
    # New replicas chosen automatically by ring
    
    # No data loss: Other replicas still available
```

**3. Re-replication (Background)**
```python
def background_re_replication():
    """Restore replication factor after node failure"""
    
    # Find under-replicated keys
    for key in all_keys():
        current_replicas = count_replicas(key)
        
        if current_replicas < replication_factor:
            # Find additional replica node
            new_replica = ring.get_replica_nodes(key, replication_factor)[-1]
            
            # Copy data from existing replica
            value = read_from_any_replica(key)
            write_to_node(new_replica, key, value)
```

**Example Timeline:**
```
t=0: Node A fails (replication=3, nodes {A,B,C})
t=1: Health monitor detects failure (15 seconds)
t=2: Remove A from ring
t=3: Reads route to B,C (no data loss)
t=4: Background re-replication to node D (5 minutes)
Final: Replication restored {B,C,D}
```

**Data Loss Scenarios:**

| Scenario | Replication | Result |
|----------|-------------|--------|
| 1 node fails | 3 replicas | No loss ✅ |
| 2 nodes fail | 3 replicas | No loss ✅ |
| 3 nodes fail | 3 replicas | Data loss ❌ |
| Network partition | Depends on quorum | May have stale reads |

**Prevention:**
- **High replication factor**: 5+ for critical data
- **Cross-datacenter replication**: Geographic diversity
- **Backup to persistent storage**: Periodic snapshots

---

### Q6: How would you implement TTL-based expiration efficiently?

**Answer:**

**Two approaches:**

**1. Lazy Expiration (our implementation)**
```python
def get(self, key: str) -> Optional[CacheEntry]:
    entry = self._storage.get(key)
    
    if entry and entry.is_expired():
        # Delete on access
        del self._storage[key]
        return None
    
    return entry

# Pros: No background threads, simple
# Cons: Expired entries consume memory until accessed
```

**2. Active Expiration (Background Cleanup)**
```python
class ActiveExpirationCache:
    def __init__(self):
        self._storage: Dict[str, CacheEntry] = {}
        self._expiry_heap: List[Tuple[float, str]] = []  # (expiry_time, key)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def put(self, entry: CacheEntry):
        self._storage[entry.key] = entry
        
        if entry.ttl_seconds:
            expiry_time = entry.timestamp + entry.ttl_seconds
            heapq.heappush(self._expiry_heap, (expiry_time, entry.key))
    
    def _cleanup_loop(self):
        """Background thread: Delete expired entries"""
        while True:
            time.sleep(1)  # Check every second
            
            now = time.time()
            
            # Delete all expired entries
            while self._expiry_heap and self._expiry_heap[0][0] <= now:
                expiry_time, key = heapq.heappop(self._expiry_heap)
                
                # Verify still expired (not updated)
                if key in self._storage:
                    entry = self._storage[key]
                    if entry.is_expired():
                        del self._storage[key]

# Pros: Memory freed proactively
# Cons: Background thread overhead
```

**3. Hybrid Approach (Redis-style)**
```python
class HybridExpirationCache:
    def __init__(self):
        self._storage: Dict[str, CacheEntry] = {}
    
    def _sample_and_delete(self):
        """Sample random keys and delete expired (Redis approach)"""
        # Sample 20 random keys
        sampled_keys = random.sample(list(self._storage.keys()), min(20, len(self._storage)))
        
        expired_count = 0
        for key in sampled_keys:
            entry = self._storage[key]
            if entry.is_expired():
                del self._storage[key]
                expired_count += 1
        
        # If >25% expired, sample again
        if expired_count > 5:
            self._sample_and_delete()

# Redis approach:
# - Sample 20 keys every 100ms
# - If >25% expired, repeat
# - Balances memory vs CPU
```

**Comparison:**

| Approach | Memory | CPU | Latency | Best For |
|----------|--------|-----|---------|----------|
| Lazy | High | Low | Low | Read-heavy |
| Active | Low | High | Low | Memory-constrained |
| Hybrid | Medium | Medium | Low | Balanced |

**Production Recommendation:**
```python
# Combine lazy + periodic cleanup
def get(key):
    # Lazy check on read
    if entry.is_expired():
        del storage[key]
        return None

# + Background cleanup every 60 seconds
def periodic_cleanup():
    for key, entry in storage.items():
        if entry.is_expired():
            del storage[key]
```

---

### Q7: Compare distributed cache with other caching strategies (CDN, local cache, database cache).

**Answer:**

**Caching Layers:**

```
┌─────────────────────────────────────────┐
│ Client (Browser)                        │
│ └─ Browser Cache (HTTP headers)        │ ← Fastest
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ CDN (Cloudflare, CloudFront)            │
│ └─ Edge locations worldwide             │ ← Static content
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ Application Server                      │
│ └─ Local Cache (per-server)            │ ← Hot keys
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ Distributed Cache (Redis, Memcached)   │
│ └─ Cluster of cache nodes               │ ← Session data
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ Database Cache (Query cache)            │
│ └─ Database internal cache              │ ← Recently queried
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ Database (PostgreSQL, MySQL)            │
│ └─ Persistent storage                   │ ← Source of truth
└─────────────────────────────────────────┘
```

**Comparison:**

| Type | Latency | Scope | Use Case | Invalidation |
|------|---------|-------|----------|--------------|
| **Local Cache** | <1ms | Single server | Hot keys, sessions | TTL, LRU |
| **Distributed Cache** | 1-5ms | Cluster-wide | User data, API responses | TTL, explicit delete |
| **CDN** | 10-50ms | Global | Static assets (images, CSS) | HTTP headers, purge API |
| **Database Cache** | N/A | Database-internal | Query results | Automatic on writes |

**When to use each:**

**1. Local Cache (Application-level)**
```python
# Use for: Hot keys, temporary data
local_cache = LRU(capacity=1000, ttl=60)

def get_user_profile(user_id):
    # Check local first
    profile = local_cache.get(user_id)
    if profile:
        return profile
    
    # Fallback to distributed cache
    profile = redis.get(user_id)
    local_cache.put(user_id, profile)
    return profile

# Pros: Sub-millisecond latency
# Cons: Per-server duplication, invalidation complexity
```

**2. Distributed Cache (Redis/Memcached)**
```python
# Use for: Session data, user profiles, API responses
redis.setex("session:abc", 3600, session_data)  # 1-hour TTL

# Pros: Shared across servers, scalable
# Cons: Network latency (1-5ms)
```

**3. CDN**
```python
# Use for: Static assets, public content
# HTTP headers:
Cache-Control: public, max-age=31536000  # 1 year
ETag: "abc123"

# Pros: Global distribution, 10-50ms latency
# Cons: Only for static/public content
```

**4. Database Cache**
```sql
-- MySQL query cache (deprecated in 8.0)
SELECT /*! SQL_CACHE */ * FROM users WHERE id = 123;

-- Pros: Automatic management
-- Cons: Limited control, complex invalidation
```

**Multi-level caching strategy:**
```python
def get_with_multi_level_cache(key):
    # L1: Local cache (fastest)
    value = local_cache.get(key)
    if value:
        return value
    
    # L2: Distributed cache
    value = redis.get(key)
    if value:
        local_cache.put(key, value)
        return value
    
    # L3: Database
    value = db.query(key)
    redis.setex(key, 300, value)
    local_cache.put(key, value)
    return value
```

---

## Testing Strategy

### Unit Tests

```python
import unittest

class TestConsistentHashRing(unittest.TestCase):
    
    def test_add_node(self):
        ring = ConsistentHashRing(virtual_nodes_per_node=10)
        node = CacheNode("node1", "10.0.0.1", 8001)
        
        keys_affected = ring.add_node(node)
        self.assertEqual(keys_affected, 10)  # 10 virtual nodes
    
    def test_get_node(self):
        ring = ConsistentHashRing()
        node1 = CacheNode("node1", "10.0.0.1", 8001)
        node2 = CacheNode("node2", "10.0.0.2", 8002)
        
        ring.add_node(node1)
        ring.add_node(node2)
        
        # Same key always maps to same node
        primary1 = ring.get_node("user:123")
        primary2 = ring.get_node("user:123")
        self.assertEqual(primary1.node_id, primary2.node_id)
    
    def test_replica_diversity(self):
        ring = ConsistentHashRing()
        for i in range(5):
            ring.add_node(CacheNode(f"node{i}", f"10.0.0.{i}", 8000+i))
        
        replicas = ring.get_replica_nodes("key1", replication_factor=3)
        
        # All replicas should be unique physical nodes
        node_ids = [r.node_id for r in replicas]
        self.assertEqual(len(node_ids), len(set(node_ids)))

class TestDistributedCache(unittest.TestCase):
    
    def test_quorum_read(self):
        cache = DistributedCache(replication_factor=3)
        for i in range(3):
            cache.add_node(CacheNode(f"node{i}", f"10.0.0.{i}", 8000+i))
        
        # Write with QUORUM
        result = cache.put("key1", "value1", consistency=ConsistencyLevel.QUORUM)
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.replicas_written, 2)
        
        # Read with QUORUM
        result = cache.get("key1", consistency=ConsistencyLevel.QUORUM)
        self.assertEqual(result.value, "value1")
        self.assertGreaterEqual(result.replicas_read, 2)
    
    def test_version_increment(self):
        cache = DistributedCache(replication_factor=2)
        cache.add_node(CacheNode("node1", "10.0.0.1", 8001))
        cache.add_node(CacheNode("node2", "10.0.0.2", 8002))
        
        # First write
        cache.put("key1", "value1")
        result1 = cache.get("key1")
        self.assertEqual(result1.version, 1)
        
        # Second write
        cache.put("key1", "value2")
        result2 = cache.get("key1")
        self.assertEqual(result2.version, 2)
```

### Integration Tests

```python
def test_node_failure_recovery():
    """Test failover when node crashes"""
    cache = DistributedCache(replication_factor=3)
    
    # Add 5 nodes
    nodes = [CacheNode(f"node{i}", f"10.0.0.{i}", 8000+i) for i in range(5)]
    for node in nodes:
        cache.add_node(node)
    
    # Write data
    cache.put("critical_key", "important_value")
    
    # Simulate node failure
    cache.remove_node(nodes[0])
    
    # Read should still work (from replicas)
    result = cache.get("critical_key")
    assert result.success
    assert result.value == "important_value"

def test_consistency_guarantees():
    """Test read-your-writes consistency"""
    cache = DistributedCache(replication_factor=3, default_consistency=ConsistencyLevel.QUORUM)
    
    for i in range(3):
        cache.add_node(CacheNode(f"node{i}", f"10.0.0.{i}", 8000+i))
    
    # Write
    cache.put("key1", "value1")
    
    # Immediate read should see write
    result = cache.get("key1")
    assert result.value == "value1"
```

---

## Production Considerations

### 1. Monitoring

**Key Metrics:**
```python
metrics = {
    "cache_hit_rate": 0.95,        # Target: >90%
    "p99_latency_ms": 5,           # Target: <10ms
    "node_count": 100,
    "keys_per_node": 1_000_000,
    "memory_utilization": 0.75,    # Target: <80%
    "hot_keys_detected": 5,
    "node_failures_24h": 0
}
```

**Alerts:**
- Hit rate drops below 85%
- Latency p99 > 20ms
- Node failure detected
- Memory utilization > 90%
- Hot key detected (>10k req/sec)

### 2. Capacity Planning

```python
# Calculate required nodes
keys_total = 100_000_000
key_size_bytes = 1_000  # Average
replication_factor = 3

total_data = keys_total * key_size_bytes * replication_factor
# = 300 GB

node_capacity = 64 * 1024**3  # 64GB RAM
nodes_required = total_data / node_capacity / 0.8  # 80% utilization
# = 6 nodes
```

### 3. Security

- **Encryption in transit**: TLS for node-to-node communication
- **Access control**: Key-based authentication
- **Audit logging**: Track all operations

### 4. Disaster Recovery

- **Cross-datacenter replication**: Geographic diversity
- **Backup to persistent storage**: Daily snapshots
- **Runbook**: Node failure, datacenter outage procedures

---

## Summary

### Do's ✅
- Use consistent hashing for data partitioning
- Implement quorum-based reads/writes for consistency
- Use version numbers for conflict resolution
- Detect and mitigate hot keys
- Monitor hit rate, latency, and failures
- Use virtual nodes for load balancing

### Don'ts ❌
- Don't use modulo hashing (poor scalability)
- Don't skip read repair (eventual consistency suffers)
- Don't ignore hot keys (causes node overload)
- Don't forget TTL expiration (memory leaks)
- Don't use ALL consistency by default (availability impact)
- Don't skip monitoring (blind to production issues)

### Key Takeaways
1. **Consistent hashing** minimizes data movement (K/N keys vs all keys)
2. **Quorum math** (W + R > N) ensures consistency guarantees
3. **Read repair** provides eventual consistency without extra RPCs
4. **Hot key detection** prevents single-node bottlenecks
5. **Replication** provides fault tolerance (no data loss on node failure)

### Complexity Summary
| Operation | Time | Space |
|-----------|------|-------|
| Add node | O(V log N) | O(V) |
| Get node | O(log N) | O(1) |
| Get replicas | O(R log N) | O(R) |
| Read (quorum) | O(R) RPCs | O(1) |
| Write (quorum) | O(R) RPCs | O(1) |

Where: V = virtual nodes, N = total ring positions, R = replication factor

---

## Design Patterns Used

1. **Consistent Hashing**: Data partitioning with minimal remapping
2. **Quorum**: Distributed consensus for consistency
3. **Read Repair**: Lazy consistency convergence
4. **Virtual Nodes**: Load balancing across heterogeneous nodes
5. **Strategy Pattern**: Configurable consistency levels

---

*This implementation demonstrates production-grade distributed cache design with fault tolerance, consistency guarantees, and scalability considerations essential for system design interviews at FAANG companies.*
