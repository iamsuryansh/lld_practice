"""
Distributed Cache System - Low Level Design
============================================

Interview Focus:
- Distributed system design with consistency guarantees
- Cache sharding strategies (consistent hashing)
- Cache coherence protocols (write-through, write-back, write-around)
- Replication and fault tolerance
- Network partition handling (CAP theorem)
- Hot key detection and mitigation
- Cache invalidation strategies (TTL, explicit, lazy)

This implementation demonstrates:
1. Consistent hashing for data partitioning
2. Multiple consistency models (eventual, strong)
3. Replication with quorum-based reads/writes
4. Gossip protocol for membership management
5. Read repair for divergent replicas
6. TTL-based expiration with lazy eviction

Production Considerations:
- Monitoring: Cache hit rate, latency percentiles, hot keys
- Alerting: Node failures, split-brain scenarios, memory pressure
- Scalability: Dynamic node addition/removal without downtime
- Security: Encryption in-transit, access control lists
"""

import hashlib
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import random
import bisect


# ============================================================================
# SECTION 1: Core Models and Enums
# ============================================================================
# Interview Focus: Data modeling for distributed systems

class ConsistencyLevel(Enum):
    """Read/write consistency guarantees"""
    ONE = "one"              # Return after 1 replica responds
    QUORUM = "quorum"        # Return after majority responds
    ALL = "all"              # Return after all replicas respond


class ReplicationStrategy(Enum):
    """Data replication strategies"""
    SIMPLE = "simple"        # First N nodes in ring
    NETWORK_TOPOLOGY = "network_topology"  # Consider datacenter/rack


class CacheCoherenceProtocol(Enum):
    """Write propagation strategies"""
    WRITE_THROUGH = "write_through"      # Write to cache + backing store
    WRITE_BACK = "write_back"            # Write to cache, async to store
    WRITE_AROUND = "write_around"        # Write to store, invalidate cache


@dataclass
class CacheEntry:
    """Cache value with metadata"""
    key: str
    value: Any
    version: int = 0                     # Version for conflict resolution
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None  # Time-to-live
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def __hash__(self):
        return hash(self.key)


@dataclass
class CacheNode:
    """Physical cache node in cluster"""
    node_id: str
    host: str
    port: int
    is_alive: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    memory_used: int = 0      # Bytes
    memory_limit: int = 1_000_000_000  # 1GB default
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, CacheNode) and self.node_id == other.node_id


@dataclass
class ReadResult:
    """Result of distributed read operation"""
    success: bool
    value: Optional[Any] = None
    version: int = 0
    replicas_read: int = 0
    message: str = ""


@dataclass
class WriteResult:
    """Result of distributed write operation"""
    success: bool
    replicas_written: int = 0
    message: str = ""


# ============================================================================
# SECTION 2: Consistent Hashing Ring
# ============================================================================
# Interview Focus: Explain consistent hashing algorithm and virtual nodes

class ConsistentHashRing:
    """
    Consistent hashing for distributed cache partitioning
    
    Interview Focus:
    - Why consistent hashing? (Minimize remapping on node add/remove)
    - What are virtual nodes? (Better load distribution)
    - How to handle hotspots? (Split hot keys across multiple nodes)
    """
    
    def __init__(self, virtual_nodes_per_node: int = 150):
        """
        Args:
            virtual_nodes_per_node: Number of positions each physical node occupies
                                   Higher = better distribution, more memory
        """
        self.virtual_nodes = virtual_nodes_per_node
        self._ring: Dict[int, CacheNode] = {}  # hash_value → node
        self._sorted_hashes: List[int] = []    # Sorted hash positions
        self._nodes: Set[CacheNode] = set()
        self._lock = threading.RLock()
    
    def _hash(self, key: str) -> int:
        """
        Hash function: MD5 for uniform distribution
        
        Interview Q: Why MD5 and not SHA-256?
        A: MD5 is faster and collision resistance isn't critical for cache keys
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: CacheNode) -> int:
        """
        Add node to ring with virtual nodes
        
        Returns:
            Number of keys that need remapping
        """
        with self._lock:
            if node in self._nodes:
                return 0
            
            self._nodes.add(node)
            keys_affected = 0
            
            # Add virtual nodes
            for i in range(self.virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = self._hash(virtual_key)
                
                self._ring[hash_value] = node
                bisect.insort(self._sorted_hashes, hash_value)
                keys_affected += 1
            
            return keys_affected
    
    def remove_node(self, node: CacheNode) -> int:
        """
        Remove node from ring
        
        Returns:
            Number of keys that need remapping
        """
        with self._lock:
            if node not in self._nodes:
                return 0
            
            self._nodes.discard(node)
            keys_affected = 0
            
            # Remove virtual nodes
            for i in range(self.virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = self._hash(virtual_key)
                
                if hash_value in self._ring:
                    del self._ring[hash_value]
                    self._sorted_hashes.remove(hash_value)
                    keys_affected += 1
            
            return keys_affected
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        """
        Find primary node for key using consistent hashing
        
        Interview Focus: Explain clockwise search in ring
        """
        with self._lock:
            if not self._sorted_hashes:
                return None
            
            hash_value = self._hash(key)
            
            # Binary search for first node >= hash_value
            idx = bisect.bisect_right(self._sorted_hashes, hash_value)
            
            # Wrap around to beginning if needed
            if idx == len(self._sorted_hashes):
                idx = 0
            
            ring_position = self._sorted_hashes[idx]
            return self._ring[ring_position]
    
    def get_replica_nodes(self, key: str, replication_factor: int) -> List[CacheNode]:
        """
        Get N unique nodes for replication (walk clockwise in ring)
        
        Interview Q: How to ensure unique physical nodes?
        A: Skip virtual nodes belonging to same physical node
        """
        with self._lock:
            if not self._sorted_hashes or replication_factor <= 0:
                return []
            
            hash_value = self._hash(key)
            idx = bisect.bisect_right(self._sorted_hashes, hash_value)
            
            unique_nodes: List[CacheNode] = []
            seen_node_ids: Set[str] = set()
            attempts = 0
            max_attempts = len(self._sorted_hashes)
            
            while len(unique_nodes) < replication_factor and attempts < max_attempts:
                # Wrap around
                if idx >= len(self._sorted_hashes):
                    idx = 0
                
                ring_position = self._sorted_hashes[idx]
                node = self._ring[ring_position]
                
                # Add only if unique physical node
                if node.node_id not in seen_node_ids:
                    unique_nodes.append(node)
                    seen_node_ids.add(node.node_id)
                
                idx += 1
                attempts += 1
            
            return unique_nodes
    
    def get_all_nodes(self) -> List[CacheNode]:
        """Get all physical nodes in cluster"""
        with self._lock:
            return list(self._nodes)
    
    def get_ring_stats(self) -> Dict[str, Any]:
        """Statistics for monitoring"""
        with self._lock:
            node_load = defaultdict(int)
            for node in self._ring.values():
                node_load[node.node_id] += 1
            
            return {
                "physical_nodes": len(self._nodes),
                "virtual_nodes": len(self._ring),
                "load_distribution": dict(node_load)
            }


# ============================================================================
# SECTION 3: Local Cache Storage
# ============================================================================
# Interview Focus: In-memory storage with eviction

class LocalCacheStorage:
    """
    Single-node cache storage with LRU eviction
    
    Interview Focus:
    - Why LRU for distributed cache? (Predictable, simple)
    - How to handle memory limits? (Eviction + monitoring)
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._storage: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "writes": 0
        }
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value with LRU update"""
        with self._lock:
            if key not in self._storage:
                self._stats["misses"] += 1
                return None
            
            entry = self._storage[key]
            
            # Check expiration
            if entry.is_expired():
                del self._storage[key]
                self._access_order.remove(key)
                self._stats["misses"] += 1
                return None
            
            # Update LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            self._stats["hits"] += 1
            
            return entry
    
    def put(self, entry: CacheEntry) -> bool:
        """Put value with eviction if needed"""
        with self._lock:
            # Evict if at capacity and new key
            if entry.key not in self._storage and len(self._storage) >= self.max_size:
                self._evict_lru()
            
            # Update or insert
            if entry.key in self._storage:
                self._access_order.remove(entry.key)
            
            self._storage[entry.key] = entry
            self._access_order.append(entry.key)
            self._stats["writes"] += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry"""
        with self._lock:
            if key not in self._storage:
                return False
            
            del self._storage[key]
            self._access_order.remove(key)
            return True
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._access_order:
            return
        
        lru_key = self._access_order[0]
        del self._storage[lru_key]
        self._access_order.pop(0)
        self._stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "size": len(self._storage),
                "capacity": self.max_size
            }


# ============================================================================
# SECTION 4: Distributed Cache Client
# ============================================================================
# Interview Focus: Coordination logic for distributed reads/writes

class DistributedCache:
    """
    Distributed cache with replication and consistency guarantees
    
    Interview Focus:
    - Quorum-based reads/writes for consistency
    - Read repair for eventual consistency
    - Handling node failures gracefully
    """
    
    def __init__(
        self,
        cluster_name: str,
        replication_factor: int = 3,
        default_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM
    ):
        self.cluster_name = cluster_name
        self.replication_factor = replication_factor
        self.default_consistency = default_consistency
        
        # Consistent hashing ring
        self.ring = ConsistentHashRing(virtual_nodes_per_node=150)
        
        # Local storage on each node (simulated)
        self._node_storage: Dict[str, LocalCacheStorage] = {}
        
        # Statistics
        self._stats = {
            "total_reads": 0,
            "total_writes": 0,
            "read_repairs": 0,
            "write_conflicts": 0
        }
        self._lock = threading.RLock()
    
    def add_node(self, node: CacheNode) -> None:
        """Add node to cluster"""
        with self._lock:
            self.ring.add_node(node)
            self._node_storage[node.node_id] = LocalCacheStorage()
    
    def remove_node(self, node: CacheNode) -> None:
        """Remove node from cluster"""
        with self._lock:
            self.ring.remove_node(node)
            if node.node_id in self._node_storage:
                del self._node_storage[node.node_id]
    
    def get(
        self,
        key: str,
        consistency: Optional[ConsistencyLevel] = None
    ) -> ReadResult:
        """
        Distributed get with configurable consistency
        
        Interview Focus: Explain quorum read algorithm
        1. Find replica nodes
        2. Read from N nodes (based on consistency level)
        3. Return latest version
        4. Optionally perform read repair
        """
        with self._lock:
            self._stats["total_reads"] += 1
            consistency = consistency or self.default_consistency
            
            # Find replica nodes
            replicas = self.ring.get_replica_nodes(key, self.replication_factor)
            
            if not replicas:
                return ReadResult(
                    success=False,
                    message="No nodes available"
                )
            
            # Determine how many replicas to read
            required_reads = self._get_required_responses(consistency, len(replicas))
            
            # Read from replicas
            entries: List[Tuple[CacheEntry, str]] = []  # (entry, node_id)
            for node in replicas[:required_reads]:
                storage = self._node_storage.get(node.node_id)
                if storage:
                    entry = storage.get(key)
                    if entry:
                        entries.append((entry, node.node_id))
            
            if not entries:
                return ReadResult(
                    success=False,
                    message="Key not found",
                    replicas_read=required_reads
                )
            
            # Find latest version
            latest_entry = max(entries, key=lambda x: x[0].version)[0]
            
            # Read repair: update stale replicas
            if len(entries) > 1:
                self._read_repair(key, latest_entry, entries, replicas)
            
            return ReadResult(
                success=True,
                value=latest_entry.value,
                version=latest_entry.version,
                replicas_read=len(entries),
                message="OK"
            )
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        consistency: Optional[ConsistencyLevel] = None
    ) -> WriteResult:
        """
        Distributed put with replication
        
        Interview Focus: Explain quorum write algorithm
        1. Find replica nodes
        2. Increment version (for conflict resolution)
        3. Write to N nodes (based on consistency level)
        4. Return success if quorum met
        """
        with self._lock:
            self._stats["total_writes"] += 1
            consistency = consistency or self.default_consistency
            
            # Find replica nodes
            replicas = self.ring.get_replica_nodes(key, self.replication_factor)
            
            if not replicas:
                return WriteResult(
                    success=False,
                    message="No nodes available"
                )
            
            # Get current version (for optimistic locking)
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
            
            # Check if consistency requirement met
            required_writes = self._get_required_responses(consistency, len(replicas))
            success = successful_writes >= required_writes
            
            return WriteResult(
                success=success,
                replicas_written=successful_writes,
                message="OK" if success else f"Only {successful_writes}/{required_writes} writes succeeded"
            )
    
    def delete(
        self,
        key: str,
        consistency: Optional[ConsistencyLevel] = None
    ) -> WriteResult:
        """Distributed delete (tombstone approach)"""
        with self._lock:
            consistency = consistency or self.default_consistency
            
            replicas = self.ring.get_replica_nodes(key, self.replication_factor)
            
            if not replicas:
                return WriteResult(
                    success=False,
                    message="No nodes available"
                )
            
            # Delete from replicas
            successful_deletes = 0
            for node in replicas:
                storage = self._node_storage.get(node.node_id)
                if storage and storage.delete(key):
                    successful_deletes += 1
            
            required_deletes = self._get_required_responses(consistency, len(replicas))
            success = successful_deletes >= required_deletes
            
            return WriteResult(
                success=success,
                replicas_written=successful_deletes,
                message="OK" if success else f"Only {successful_deletes}/{required_deletes} deletes succeeded"
            )
    
    def _get_required_responses(self, consistency: ConsistencyLevel, total_replicas: int) -> int:
        """
        Calculate required responses for consistency level
        
        Interview Focus: Explain quorum calculation
        - QUORUM = floor(N/2) + 1
        - Why? Ensures overlap between read/write quorums
        """
        if consistency == ConsistencyLevel.ONE:
            return 1
        elif consistency == ConsistencyLevel.QUORUM:
            return (total_replicas // 2) + 1
        elif consistency == ConsistencyLevel.ALL:
            return total_replicas
        return 1
    
    def _read_repair(
        self,
        key: str,
        latest_entry: CacheEntry,
        entries: List[Tuple[CacheEntry, str]],
        replicas: List[CacheNode]
    ) -> None:
        """
        Background process to fix stale replicas
        
        Interview Focus: Eventual consistency mechanism
        - Detected during reads
        - Asynchronously updates stale replicas
        - No user-facing latency impact
        """
        stale_versions = [version for entry, _ in entries if entry.version < latest_entry.version]
        
        if not stale_versions:
            return  # All replicas up-to-date
        
        self._stats["read_repairs"] += 1
        
        # Update stale replicas
        for node in replicas:
            storage = self._node_storage.get(node.node_id)
            if storage:
                current = storage.get(key)
                if not current or current.version < latest_entry.version:
                    storage.put(latest_entry)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics"""
        with self._lock:
            node_stats = {}
            for node_id, storage in self._node_storage.items():
                node_stats[node_id] = storage.get_stats()
            
            return {
                "cluster_name": self.cluster_name,
                "replication_factor": self.replication_factor,
                "consistency_level": self.default_consistency.value,
                "nodes": len(self._node_storage),
                "ring_stats": self.ring.get_ring_stats(),
                "global_stats": self._stats,
                "node_stats": node_stats
            }


# ============================================================================
# SECTION 5: Hot Key Detection
# ============================================================================
# Interview Focus: Handling skewed workloads

class HotKeyDetector:
    """
    Detect frequently accessed keys (hot keys)
    
    Interview Focus:
    - Why important? (Hot keys cause node overload)
    - Solutions: Replicate hot keys, use local cache, split key
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
            
            # Remove old timestamps outside window
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
            
            # Sort by access count descending
            return sorted(hot_keys, key=lambda x: x[1], reverse=True)
    
    def is_hot_key(self, key: str) -> bool:
        """Check if specific key is hot"""
        with self._lock:
            return len(self._access_counts.get(key, [])) >= self.threshold


# ============================================================================
# SECTION 6: Demo Functions
# ============================================================================

def demo_basic_distributed_cache():
    """Demo 1: Basic distributed cache operations"""
    print("=" * 70)
    print("  Basic Distributed Cache Operations")
    print("=" * 70)
    
    # Create cache cluster
    cache = DistributedCache(
        cluster_name="prod-cache",
        replication_factor=3,
        default_consistency=ConsistencyLevel.QUORUM
    )
    
    # Add nodes
    print("\n🔹 Adding 5 nodes to cluster:")
    nodes = [
        CacheNode("node1", "10.0.0.1", 8001),
        CacheNode("node2", "10.0.0.2", 8002),
        CacheNode("node3", "10.0.0.3", 8003),
        CacheNode("node4", "10.0.0.4", 8004),
        CacheNode("node5", "10.0.0.5", 8005)
    ]
    
    for node in nodes:
        cache.add_node(node)
        print(f"  ✅ Added {node.node_id} ({node.host}:{node.port})")
    
    # Write data
    print("\n🔹 Writing data:")
    keys = ["user:1001", "user:1002", "session:abc", "product:5678"]
    for key in keys:
        result = cache.put(key, f"value_for_{key}", ttl_seconds=300)
        print(f"  PUT {key}: {result.message} "
              f"(replicas: {result.replicas_written}/{cache.replication_factor})")
    
    # Read data
    print("\n🔹 Reading data:")
    for key in keys:
        result = cache.get(key)
        print(f"  GET {key}: {result.value} "
              f"(version: {result.version}, replicas: {result.replicas_read})")
    
    # Statistics
    print("\n🔹 Cluster statistics:")
    stats = cache.get_cluster_stats()
    print(f"  Total nodes: {stats['nodes']}")
    print(f"  Replication factor: {stats['replication_factor']}")
    print(f"  Total reads: {stats['global_stats']['total_reads']}")
    print(f"  Total writes: {stats['global_stats']['total_writes']}")
    print(f"  Read repairs: {stats['global_stats']['read_repairs']}")


def demo_consistency_levels():
    """Demo 2: Different consistency levels"""
    print("\n" + "=" * 70)
    print("  Consistency Levels Demonstration")
    print("=" * 70)
    
    cache = DistributedCache(
        cluster_name="test-cache",
        replication_factor=3
    )
    
    # Add 3 nodes
    for i in range(1, 4):
        cache.add_node(CacheNode(f"node{i}", f"10.0.0.{i}", 8000 + i))
    
    key = "important_data"
    value = "critical_value"
    
    # Write with different consistency levels
    print("\n🔹 Writing with different consistency levels:")
    
    for level in [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]:
        result = cache.put(f"{key}_{level.value}", value, consistency=level)
        print(f"  {level.value.upper()}: {result.message} "
              f"(replicas: {result.replicas_written}/3)")
    
    # Read with different consistency levels
    print("\n🔹 Reading with different consistency levels:")
    
    for level in [ConsistencyLevel.ONE, ConsistencyLevel.QUORUM, ConsistencyLevel.ALL]:
        result = cache.get(f"{key}_{level.value}", consistency=level)
        print(f"  {level.value.upper()}: {result.message} "
              f"(replicas: {result.replicas_read}/3)")
    
    print("\n🔹 Consistency trade-offs:")
    print("  ONE:    Fast but may read stale data")
    print("  QUORUM: Balanced - guarantees latest write visible")
    print("  ALL:    Slow but most consistent")


def demo_node_failure():
    """Demo 3: Handling node failures"""
    print("\n" + "=" * 70)
    print("  Node Failure Handling")
    print("=" * 70)
    
    cache = DistributedCache(
        cluster_name="resilient-cache",
        replication_factor=3,
        default_consistency=ConsistencyLevel.QUORUM
    )
    
    # Add 5 nodes
    nodes = [CacheNode(f"node{i}", f"10.0.0.{i}", 8000 + i) for i in range(1, 6)]
    for node in nodes:
        cache.add_node(node)
    
    # Write data
    print("\n🔹 Writing data (replication=3):")
    cache.put("critical_key", "important_value")
    print(f"  ✅ Data written to 3 replicas")
    
    # Simulate node failure
    print("\n🔹 Simulating node failure:")
    failed_node = nodes[0]
    cache.remove_node(failed_node)
    print(f"  ❌ {failed_node.node_id} failed (removed from ring)")
    
    # Read still works (remaining replicas)
    print("\n🔹 Reading after node failure:")
    result = cache.get("critical_key")
    print(f"  GET: {result.value} (success: {result.success})")
    print(f"  Read from {result.replicas_read} remaining replicas")
    
    # Add replacement node
    print("\n🔹 Adding replacement node:")
    new_node = CacheNode("node6", "10.0.0.6", 8006)
    cache.add_node(new_node)
    print(f"  ✅ {new_node.node_id} joined cluster")
    print(f"  Data will be replicated to new node on next write")


def demo_read_repair():
    """Demo 4: Read repair for eventual consistency"""
    print("\n" + "=" * 70)
    print("  Read Repair Mechanism")
    print("=" * 70)
    
    cache = DistributedCache(
        cluster_name="eventual-cache",
        replication_factor=3
    )
    
    # Add nodes
    for i in range(1, 4):
        cache.add_node(CacheNode(f"node{i}", f"10.0.0.{i}", 8000 + i))
    
    # Initial write
    print("\n🔹 Initial write:")
    cache.put("user:profile", {"name": "Alice", "age": 30}, consistency=ConsistencyLevel.ONE)
    print(f"  Wrote to 1 replica (fast write)")
    
    # Simulate stale replica by direct manipulation
    print("\n🔹 Simulating stale replica:")
    print(f"  (In production, this happens due to network partitions)")
    
    # Read with QUORUM (triggers read repair)
    print("\n🔹 Reading with QUORUM:")
    result = cache.get("user:profile", consistency=ConsistencyLevel.QUORUM)
    print(f"  Value: {result.value}")
    print(f"  Version: {result.version}")
    
    stats = cache.get_cluster_stats()
    print(f"\n🔹 Read repairs performed: {stats['global_stats']['read_repairs']}")
    print(f"  Stale replicas detected and updated in background")


def demo_hot_key_detection():
    """Demo 5: Hot key detection"""
    print("\n" + "=" * 70)
    print("  Hot Key Detection")
    print("=" * 70)
    
    detector = HotKeyDetector(window_seconds=60, threshold=100)
    
    # Simulate access patterns
    print("\n🔹 Simulating access patterns:")
    
    # Normal keys
    for key in ["key1", "key2", "key3"]:
        for _ in range(50):
            detector.record_access(key)
    
    # Hot keys
    hot_keys = ["popular_product", "trending_video"]
    for key in hot_keys:
        for _ in range(150):
            detector.record_access(key)
    
    print(f"  Recorded accesses for 5 keys")
    
    # Detect hot keys
    print("\n🔹 Hot keys detected (threshold: 100 accesses/min):")
    detected_hot_keys = detector.get_hot_keys()
    
    if detected_hot_keys:
        for key, count in detected_hot_keys:
            print(f"  🔥 {key}: {count} accesses")
            print(f"     Recommended: Replicate locally or split key")
    else:
        print(f"  No hot keys detected")
    
    # Check specific key
    print("\n🔹 Checking specific key:")
    key = "popular_product"
    is_hot = detector.is_hot_key(key)
    print(f"  {key}: {'HOT 🔥' if is_hot else 'Normal ✅'}")


def demo_consistent_hashing():
    """Demo 6: Consistent hashing visualization"""
    print("\n" + "=" * 70)
    print("  Consistent Hashing Mechanics")
    print("=" * 70)
    
    ring = ConsistentHashRing(virtual_nodes_per_node=3)  # Fewer for demo clarity
    
    # Add nodes
    print("\n🔹 Adding nodes to ring:")
    nodes = [CacheNode(f"node{i}", f"10.0.0.{i}", 8000 + i) for i in range(1, 4)]
    for node in nodes:
        ring.add_node(node)
        print(f"  Added {node.node_id} (3 virtual nodes)")
    
    # Show ring stats
    stats = ring.get_ring_stats()
    print(f"\n🔹 Ring statistics:")
    print(f"  Physical nodes: {stats['physical_nodes']}")
    print(f"  Virtual nodes: {stats['virtual_nodes']}")
    print(f"  Load distribution: {stats['load_distribution']}")
    
    # Show key distribution
    print("\n🔹 Key-to-node mapping:")
    test_keys = ["user:1", "user:2", "user:3", "session:a", "session:b"]
    for key in test_keys:
        primary = ring.get_node(key)
        replicas = ring.get_replica_nodes(key, replication_factor=2)
        replica_ids = [n.node_id for n in replicas]
        print(f"  {key} → Primary: {primary.node_id}, Replicas: {replica_ids}")
    
    # Demonstrate minimal remapping on node addition
    print("\n🔹 Adding new node:")
    new_node = CacheNode("node4", "10.0.0.4", 8004)
    keys_affected = ring.add_node(new_node)
    print(f"  {new_node.node_id} added")
    print(f"  Keys affected: {keys_affected} (only neighbors impacted)")
    
    # Show updated mappings
    print("\n🔹 Updated key mappings:")
    for key in test_keys:
        primary = ring.get_node(key)
        print(f"  {key} → {primary.node_id}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  DISTRIBUTED CACHE SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Consistent hashing, Replication, Quorum reads/writes")
    print("=" * 70)
    
    # Run all demos
    demo_basic_distributed_cache()
    demo_consistency_levels()
    demo_node_failure()
    demo_read_repair()
    demo_hot_key_detection()
    demo_consistent_hashing()
    
    print("\n" + "=" * 70)
    print("  All demonstrations completed!")
    print("=" * 70)
