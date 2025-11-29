# Distributed Key-Value Store - Low Level Design

## Problem Statement

Design a **production-grade distributed key-value store** similar to Amazon Dynamo, Apache Cassandra, or Riak. The system must handle:

1. **High availability**: Continue operating during node failures
2. **Eventual consistency**: Trade-off between consistency and availability (CAP theorem)
3. **Scalability**: Handle billions of keys across hundreds of nodes
4. **Conflict resolution**: Handle concurrent writes from multiple clients
5. **Data durability**: Survive crashes without data loss
6. **Efficient synchronization**: Detect and repair inconsistencies quickly

### Real-World Context
Used by: Amazon (DynamoDB), LinkedIn (Voldemort), Discord, Apple (iCloud), Netflix

### Key Requirements
- **Availability**: 99.99% uptime even during failures
- **Performance**: <10ms read/write latency at p99
- **Scalability**: Linear scale-out by adding nodes
- **Consistency**: Tunable (strong vs eventual)
- **Durability**: Zero data loss on single node failure

---

## Implementation Phases

### Phase 1: Vector Clocks for Causality (15-20 minutes)

**Core concept**: Track causality between events to detect conflicts

```python
class VectorClock:
    """
    Vector clock for causality tracking
    
    Key insight: Lamport timestamps show total order, but we need
    to detect CONCURRENT events (neither happened-before the other)
    """
    
    def __init__(self):
        self._clocks: Dict[str, int] = {}  # node_id -> counter
    
    def increment(self, node_id: str) -> None:
        """Increment counter for this node"""
        self._clocks[node_id] = self._clocks.get(node_id, 0) + 1
    
    def merge(self, other: 'VectorClock') -> None:
        """Merge two vector clocks (take max of each entry)"""
        for node_id, counter in other._clocks.items():
            self._clocks[node_id] = max(
                self._clocks.get(node_id, 0),
                counter
            )
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """
        Check if self happened before other
        
        self -> other if:
        - All entries in self <= other
        - At least one entry in self < other
        """
        all_less_or_equal = all(
            self._clocks.get(k, 0) <= other._clocks.get(k, 0)
            for k in set(self._clocks) | set(other._clocks)
        )
        
        some_less = any(
            self._clocks.get(k, 0) < other._clocks.get(k, 0)
            for k in set(self._clocks) | set(other._clocks)
        )
        
        return all_less_or_equal and some_less
    
    def concurrent(self, other: 'VectorClock') -> bool:
        """Check if two events are concurrent (neither happened before)"""
        return not self.happens_before(other) and not other.happens_before(self)
```

**Interview questions to expect**:
- Q: "Why not use timestamps?"
- A: "Wall-clock timestamps can't detect causality in distributed systems due to clock skew. Vector clocks track per-node counters to establish happens-before relationships."

- Q: "What's the space overhead?"
- A: "O(N) where N = number of nodes. For 100 nodes, ~800 bytes per version. Can prune with techniques like dotted version vectors."

---

### Phase 2: Merkle Trees for Anti-Entropy (20-25 minutes)

**Core concept**: Efficiently detect divergence between replicas

```python
@dataclass
class MerkleNode:
    """Node in Merkle tree"""
    hash_value: str
    start_key: str
    end_key: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    
class MerkleTree:
    """
    Merkle tree for efficient replica comparison
    
    Key insight: Compare O(log N) hashes instead of O(N) keys
    """
    
    def __init__(self, key_range: Tuple[str, str]):
        self.key_range = key_range
        self.root: Optional[MerkleNode] = None
    
    def build(self, data: Dict[str, bytes]) -> None:
        """Build tree from key-value data"""
        sorted_keys = sorted(data.keys())
        self.root = self._build_recursive(sorted_keys, data, 0, len(sorted_keys))
    
    def _build_recursive(
        self,
        keys: List[str],
        data: Dict[str, bytes],
        start: int,
        end: int
    ) -> Optional[MerkleNode]:
        """Recursively build tree"""
        if start >= end:
            return None
        
        if end - start == 1:
            # Leaf node
            key = keys[start]
            hash_value = hashlib.sha256(data[key]).hexdigest()
            return MerkleNode(hash_value, key, key)
        
        # Internal node
        mid = (start + end) // 2
        left = self._build_recursive(keys, data, start, mid)
        right = self._build_recursive(keys, data, mid, end)
        
        # Combine hashes
        combined = (left.hash_value if left else "") + (right.hash_value if right else "")
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        return MerkleNode(
            hash_value=hash_value,
            start_key=keys[start],
            end_key=keys[end - 1],
            left=left,
            right=right
        )
    
    def compare(self, other: 'MerkleTree') -> List[Tuple[str, str]]:
        """
        Compare with another tree, return divergent key ranges
        
        Returns: List of (start_key, end_key) ranges that differ
        """
        divergent = []
        self._compare_recursive(self.root, other.root, divergent)
        return divergent
    
    def _compare_recursive(
        self,
        node1: Optional[MerkleNode],
        node2: Optional[MerkleNode],
        divergent: List[Tuple[str, str]]
    ) -> None:
        """Recursively compare trees"""
        if not node1 or not node2:
            return
        
        if node1.hash_value == node2.hash_value:
            # Subtrees are identical
            return
        
        if not node1.left and not node1.right:
            # Leaf nodes differ
            divergent.append((node1.start_key, node1.end_key))
            return
        
        # Recurse on children
        self._compare_recursive(node1.left, node2.left, divergent)
        self._compare_recursive(node1.right, node2.right, divergent)
```

**Interview insight**:
- Without Merkle trees: Compare ALL keys → O(N) time, O(N) network
- With Merkle trees: Compare hashes → O(log N) time, O(divergent) network
- Real-world: 99.9% of syncs find NO divergence → 1 hash comparison vs millions

---

### Phase 3: Write-Ahead Log for Durability (20-25 minutes)

**Core concept**: Persist operations before applying them

```python
class WriteAheadLog:
    """
    WAL for crash recovery
    
    Key insight: Write to sequential log BEFORE applying to in-memory state
    This ensures durability with minimal performance impact
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._lock = threading.Lock()
    
    def append(self, operation: str, key: str, value: Any, vector_clock: VectorClock) -> None:
        """Append operation to log"""
        with self._lock:
            entry = {
                "timestamp": time.time(),
                "operation": operation,  # "PUT", "DELETE"
                "key": key,
                "value": value,
                "vector_clock": vector_clock.to_dict()
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
                f.flush()  # Force to disk
                os.fsync(f.fileno())  # Ensure disk write
    
    def replay(self) -> List[Dict]:
        """Replay log to reconstruct state after crash"""
        operations = []
        
        if not os.path.exists(self.log_file):
            return operations
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    operations.append(json.loads(line))
        
        return operations
    
    def compact(self, active_keys: Set[str]) -> None:
        """
        Compact log by removing obsolete entries
        
        Strategy: Keep only latest operation per key
        """
        latest_ops = {}
        
        # Find latest operation per key
        for op in self.replay():
            latest_ops[op['key']] = op
        
        # Write compacted log
        temp_file = self.log_file + '.tmp'
        with open(temp_file, 'w') as f:
            for key in active_keys:
                if key in latest_ops:
                    f.write(json.dumps(latest_ops[key]) + '\n')
        
        os.replace(temp_file, self.log_file)
```

**Interview discussion**:
- Q: "Why not write directly to disk-based B-tree?"
- A: "Sequential writes (WAL) are 10-100x faster than random writes (B-tree). WAL gives ~1ms write latency vs ~10-50ms for direct B-tree updates."

- Q: "When to compact?"
- A: "When log size exceeds threshold (e.g., 10x active data) OR periodically (e.g., every hour). Trade-off: compaction I/O vs log size."

---

### Phase 4: Quorum Consensus (25-30 minutes)

**Core concept**: Tunable consistency with R + W > N

```python
class DistributedKVStore:
    """
    Distributed KV store with quorum consensus
    
    Tunable consistency:
    - N: Replication factor (default: 3)
    - R: Read quorum (default: 2)
    - W: Write quorum (default: 2)
    
    Guarantee: R + W > N ensures read-after-write consistency
    """
    
    def __init__(self, cluster_name: str, replication_factor: int = 3):
        self.cluster_name = cluster_name
        self.replication_factor = replication_factor
        self.consistent_hash = ConsistentHashRing()
        self.nodes: Dict[str, LocalStorageEngine] = {}
        self._vector_clocks: Dict[str, VectorClock] = {}
    
    def put(
        self,
        key: str,
        value: Any,
        write_quorum: int = 2
    ) -> bool:
        """
        Write with quorum
        
        Steps:
        1. Hash key to find preference list (N replicas)
        2. Increment vector clock
        3. Write to W replicas in parallel
        4. Return success if W writes succeed
        """
        # Get preference list (N nodes)
        preference_list = self._get_preference_list(key, self.replication_factor)
        
        if len(preference_list) < write_quorum:
            return False
        
        # Increment vector clock
        if key not in self._vector_clocks:
            self._vector_clocks[key] = VectorClock()
        
        self._vector_clocks[key].increment(preference_list[0])
        
        # Write to replicas
        successful_writes = 0
        for node_id in preference_list:
            if node_id in self.nodes:
                try:
                    self.nodes[node_id].put(
                        key,
                        value,
                        self._vector_clocks[key].copy()
                    )
                    successful_writes += 1
                    
                    if successful_writes >= write_quorum:
                        return True
                except Exception:
                    continue
        
        return successful_writes >= write_quorum
    
    def get(
        self,
        key: str,
        read_quorum: int = 2
    ) -> Optional[Any]:
        """
        Read with quorum
        
        Steps:
        1. Read from R replicas in parallel
        2. Reconcile versions using vector clocks
        3. Perform read repair for stale replicas
        4. Return latest version
        """
        preference_list = self._get_preference_list(key, self.replication_factor)
        
        # Read from replicas
        versions = []
        for node_id in preference_list[:read_quorum]:
            if node_id in self.nodes:
                result = self.nodes[node_id].get(key)
                if result:
                    versions.append((node_id, result))
        
        if not versions:
            return None
        
        # Reconcile versions
        latest = self._reconcile_versions(versions)
        
        # Read repair (async in production)
        self._read_repair(key, latest, preference_list)
        
        return latest.value
    
    def _reconcile_versions(
        self,
        versions: List[Tuple[str, 'Versioned']]
    ) -> 'Versioned':
        """
        Reconcile conflicting versions
        
        Strategy:
        1. If one version happened-before another, use later
        2. If concurrent, use application-level resolver (e.g., LWW)
        """
        if len(versions) == 1:
            return versions[0][1]
        
        # Find latest based on vector clocks
        latest = versions[0][1]
        
        for _, version in versions[1:]:
            if version.vector_clock.happens_before(latest.vector_clock):
                continue  # latest is newer
            elif latest.vector_clock.happens_before(version.vector_clock):
                latest = version  # version is newer
            else:
                # Concurrent - use last-write-wins
                if version.timestamp > latest.timestamp:
                    latest = version
        
        return latest
```

**Interview deep-dive**:

**Q: "Explain why R + W > N guarantees consistency"**

A: "Mathematical proof:
- N = 3 replicas: {A, B, C}
- R = 2, W = 2, R + W = 4 > 3
- Write updates 2 of {A, B, C}
- Read queries 2 of {A, B, C}
- Pigeonhole principle: Read MUST overlap with write
- At least 1 replica in read set has latest version
- Reconciliation selects latest → consistency guaranteed"

**Q: "What if you want stronger consistency?"**

A: "Options:
1. **R = N, W = 1**: All nodes must respond to reads (strong consistency, low availability)
2. **R = 1, W = N**: All writes go to all nodes (strong consistency for reads, slow writes)
3. **R = W = QUORUM**: Balanced (QUORUM = N/2 + 1)"

**Q: "Trade-offs?"**

A: "CAP theorem in action:
- High R, High W → Consistency, but low availability (can't tolerate many failures)
- Low R, Low W → Availability, but eventual consistency
- Production: N=3, R=2, W=2 (tolerate 1 failure, strong consistency)"

---

### Phase 5: Anti-Entropy with Merkle Trees (25-30 minutes)

**Core concept**: Background process to fix inconsistencies

```python
def anti_entropy_sync(self, node1_id: str, node2_id: str) -> List[str]:
    """
    Synchronize two nodes using Merkle trees
    
    Process:
    1. Both nodes build Merkle trees
    2. Compare trees to find divergent key ranges
    3. Exchange only divergent keys
    4. Reconcile and update
    """
    node1 = self.nodes.get(node1_id)
    node2 = self.nodes.get(node2_id)
    
    if not node1 or not node2:
        return []
    
    # Build Merkle trees
    tree1 = node1.build_merkle_tree()
    tree2 = node2.build_merkle_tree()
    
    # Find divergent ranges
    divergent_ranges = tree1.compare(tree2)
    
    if not divergent_ranges:
        return []  # Already in sync
    
    # Exchange divergent keys
    synced_keys = []
    for start_key, end_key in divergent_ranges:
        # Get versions from both nodes
        keys = self._get_keys_in_range(node1, start_key, end_key)
        
        for key in keys:
            v1 = node1.get(key)
            v2 = node2.get(key)
            
            if not v1 and v2:
                node1.put(key, v2.value, v2.vector_clock)
                synced_keys.append(key)
            elif v1 and not v2:
                node2.put(key, v1.value, v1.vector_clock)
                synced_keys.append(key)
            elif v1 and v2:
                # Reconcile
                if v1.vector_clock.happens_before(v2.vector_clock):
                    node1.put(key, v2.value, v2.vector_clock)
                    synced_keys.append(key)
                elif v2.vector_clock.happens_before(v1.vector_clock):
                    node2.put(key, v1.value, v1.vector_clock)
                    synced_keys.append(key)
    
    return synced_keys
```

**Interview insight**:
- **Without Merkle trees**: Compare all keys → 1M keys = 1M comparisons
- **With Merkle trees**: Compare tree hashes → 1M keys = ~20 hash comparisons
- **Efficiency**: 50,000x fewer operations for in-sync replicas

---

## Critical Knowledge Points

### 1. Vector Clocks vs Timestamps

**Timestamps (broken)**:
```
Node A: Write X at 10:00:01
Node B: Write X at 10:00:00 (but clock is fast by 2 seconds)

With timestamps: B's write (10:00:00) < A's write (10:00:01) → Wrong!
Reality: B's write actually happened AFTER A's write
```

**Vector Clocks (correct)**:
```
Node A: Write X → VC = {A: 1, B: 0}
Node B: Write X → VC = {A: 0, B: 1}

Concurrent (neither happened-before) → Conflict detected correctly
```

### 2. Merkle Tree Efficiency

**Sync without Merkle trees**:
- Node A: 1,000,000 keys
- Node B: 1,000,000 keys (999,999 same, 1 different)
- Network: Transfer 1M keys → ~100 MB
- Time: ~10 seconds

**Sync with Merkle trees**:
- Compare root hash → different
- Compare children → find divergent subtree
- Drill down: log₂(1M) ≈ 20 levels
- Network: Transfer 1 key → ~1 KB
- Time: ~100ms

### 3. Quorum Math Examples

**Scenario 1: N=3, R=1, W=1**
- Read from 1 node, write to 1 node
- R + W = 2 ≤ N = 3 → NO GUARANTEE
- Possible: Read stale data (read from node that wasn't written to)

**Scenario 2: N=3, R=2, W=2**
- Read from 2 nodes, write to 2 nodes
- R + W = 4 > N = 3 → GUARANTEED OVERLAP
- Impossible: Read must see at least 1 updated node

**Scenario 3: N=5, R=3, W=3**
- Read from 3 nodes, write to 3 nodes
- R + W = 6 > N = 5 → GUARANTEED
- Tolerates 2 failures: Min(R, W) - 1 = 2

---

## Interview Q&A

### Q1: "How does this compare to traditional databases?"

**Answer**: 

| Aspect | RDBMS (MySQL) | Distributed KV (Dynamo) |
|--------|---------------|-------------------------|
| **Consistency** | Strong (ACID) | Tunable (eventual to strong) |
| **Availability** | Single-master (lower) | Multi-master (higher) |
| **Scalability** | Vertical (expensive) | Horizontal (cheap) |
| **Latency** | ~10-50ms | ~1-10ms |
| **Query** | Complex SQL | Simple get/put |
| **Use case** | Transactions, analytics | High-throughput, low-latency |

**Trade-off**: Distributed KV sacrifices query flexibility for availability and performance.

---

### Q2: "How to handle network partitions?"

**Answer**: Depends on CAP theorem choice:

**AP (Available + Partition-tolerant)**: Like Dynamo
```
- Allow writes to both sides of partition
- Vector clocks detect conflicts
- Reconcile when partition heals
- Trade-off: Temporary inconsistency
```

**CP (Consistent + Partition-tolerant)**: Like HBase
```
- Reject writes to minority partition
- Only majority partition stays available
- Trade-off: Reduced availability during partition
```

**Interview tip**: "There's no perfect choice. AP is better for shopping carts (merge conflicts OK), CP is better for banking (consistency critical)."

---

### Q3: "Explain write amplification in LSM trees"

**Answer**: 

LSM-trees (used in storage engines like RocksDB, LevelDB):
```
Write path:
1. Write to WAL (sequential) → 1x write
2. Write to MemTable (memory) → 0x disk writes
3. Flush MemTable to L0 (SSTable) → 1x write
4. Compact L0 → L1 → 2x write (read + rewrite)
5. Compact L1 → L2 → 2x write
...

Total: ~10-50x write amplification
```

**Mitigation**:
- Larger MemTable (fewer flushes)
- Leveled compaction (less rewriting)
- Tiered compaction (for time-series data)

**Trade-off**: Write amplification for read performance (sorted, indexed SSTables)

---

### Q4: "How to handle hot keys?"

**Answer**: Multiple strategies:

**1. Increased replication**:
```python
if detect_hot_key(key):
    replicate_to_all_nodes(key)  # N = cluster_size instead of 3
```

**2. Local caching**:
```python
if access_frequency(key) > threshold:
    cache_locally(key)  # Don't hit distributed store
```

**3. Key splitting**:
```python
# Split "celebrity:123" into multiple keys
keys = ["celebrity:123:0", "celebrity:123:1", "celebrity:123:2"]
value = merge([get(k) for k in keys])
```

**4. Read-only replicas**:
```python
# Send reads to read-only replicas, writes to primary
if operation == "GET":
    route_to_read_replica(key)
```

---

### Q5: "Explain sloppy quorum and hinted handoff"

**Answer**:

**Problem**: Node A is down, can't achieve write quorum

**Traditional quorum** (strict):
```
N=3, W=2, nodes={A, B, C}
Node A is down
Write to B, C → success (2/3 = quorum)
```

**Sloppy quorum** (available):
```
N=3, W=2, nodes={A, B, C}
Node A is down
Write to D (temporary), B → success
When A recovers: D sends "hint" to A (hinted handoff)
```

**Benefits**:
- Higher availability (can tolerate more failures)
- Faster recovery (hints trigger sync)

**Trade-off**:
- Temporary inconsistency (hint might be lost)
- More complex failure handling

---

### Q6: "Compare with Cassandra and Riak"

**Answer**:

| Feature | Dynamo (AWS) | Cassandra | Riak |
|---------|--------------|-----------|------|
| **Consistency** | Vector clocks | LWW timestamps | Vector clocks |
| **Data model** | Key-value | Wide-column | Key-value |
| **Query** | Get/Put only | CQL (SQL-like) | Get/Put, MapReduce |
| **Replication** | Consistent hashing | Token rings | Consistent hashing |
| **Conflict** | App-level | Last-write-wins | App-level or CRDT |
| **Use case** | Shopping cart | Time-series, IoT | Session store, logs |

**Key difference**: Cassandra trades conflict resolution complexity for simpler LWW timestamps.

---

### Q7: "How to test a distributed KV store?"

**Answer**:

**1. Unit tests**:
```python
def test_vector_clock_causality():
    vc1 = VectorClock({"A": 1, "B": 0})
    vc2 = VectorClock({"A": 1, "B": 1})
    assert vc1.happens_before(vc2)
    assert not vc2.happens_before(vc1)
```

**2. Integration tests**:
```python
def test_quorum_consistency():
    store = DistributedKVStore(replication_factor=3)
    store.put("key", "value", write_quorum=2)
    assert store.get("key", read_quorum=2) == "value"
```

**3. Chaos testing**:
```python
def test_network_partition():
    partition_network(["node1", "node2"], ["node3"])
    store.put("key", "A")  # Write to minority
    heal_partition()
    assert_eventually_consistent("key")
```

**4. Performance tests**:
```python
def test_throughput():
    start = time.time()
    for i in range(100000):
        store.put(f"key{i}", f"value{i}")
    elapsed = time.time() - start
    assert elapsed < 10  # <10s for 100k writes
```

---

## Testing Strategy

### Unit Tests
```python
def test_vector_clock_happens_before():
    vc1 = VectorClock({"A": 1, "B": 2})
    vc2 = VectorClock({"A": 2, "B": 3})
    assert vc1.happens_before(vc2)

def test_vector_clock_concurrent():
    vc1 = VectorClock({"A": 2, "B": 1})
    vc2 = VectorClock({"A": 1, "B": 2})
    assert vc1.concurrent(vc2)

def test_merkle_tree_identical():
    tree1 = MerkleTree(("a", "z"))
    tree1.build({"key1": b"value1", "key2": b"value2"})
    
    tree2 = MerkleTree(("a", "z"))
    tree2.build({"key1": b"value1", "key2": b"value2"})
    
    assert tree1.root.hash_value == tree2.root.hash_value
    assert len(tree1.compare(tree2)) == 0

def test_wal_recovery():
    wal = WriteAheadLog("/tmp/wal.log")
    wal.append("PUT", "key1", "value1", VectorClock({"A": 1}))
    
    # Simulate crash and recovery
    recovered = wal.replay()
    assert len(recovered) == 1
    assert recovered[0]["key"] == "key1"
```

### Integration Tests
```python
def test_quorum_consistency():
    store = DistributedKVStore(cluster_name="test", replication_factor=3)
    
    # Add nodes
    for i in range(5):
        store.add_node(f"node{i}", f"10.0.0.{i}", 9000 + i)
    
    # Write with quorum
    success = store.put("user:123", {"name": "Alice"}, write_quorum=2)
    assert success
    
    # Read with quorum
    value = store.get("user:123", read_quorum=2)
    assert value["name"] == "Alice"

def test_anti_entropy_sync():
    store = DistributedKVStore(cluster_name="test")
    
    # Create divergence
    store.nodes["node1"].put("key1", "value1", VectorClock({"node1": 1}))
    store.nodes["node2"].put("key1", "value2", VectorClock({"node2": 1}))
    
    # Sync
    synced = store.anti_entropy_sync("node1", "node2")
    assert "key1" in synced
    
    # Verify reconciliation
    v1 = store.nodes["node1"].get("key1")
    v2 = store.nodes["node2"].get("key1")
    assert v1.value == v2.value
```

---

## Production Considerations

### 1. Storage Backend
- **MemTable**: Skip list or Red-Black tree for sorted in-memory data
- **SSTables**: Immutable on-disk files with Bloom filters
- **Compaction**: Level-based (less space) vs Size-tiered (less I/O)

### 2. Monitoring Metrics
```python
metrics = {
    "read_latency_p99": 5,  # milliseconds
    "write_latency_p99": 10,
    "quorum_failures": 0.01,  # percentage
    "anti_entropy_lag": 60,  # seconds
    "storage_size": 100,  # GB
    "compaction_throughput": 50  # MB/s
}
```

### 3. Operational Considerations
- **Node addition**: Use virtual nodes for gradual rebalancing
- **Node removal**: Transfer data to successors before decommission
- **Backup**: Snapshot SSTables + WAL to S3
- **Recovery**: Replay WAL + load SSTables
- **Monitoring**: Track quorum failures, repair lag, hot keys

### 4. Security
- **Encryption**: TLS for network, AES-256 for storage
- **Authentication**: Client certificates or tokens
- **Authorization**: Per-key ACLs
- **Audit**: Log all operations with vector clocks

---

## Summary

### Do's ✅
- Use vector clocks for causality tracking
- Implement Merkle trees for efficient synchronization
- Use WAL for durability (sequential writes are fast)
- Tune R, W, N based on consistency requirements
- Implement anti-entropy for eventual consistency
- Monitor quorum failures and repair lag

### Don'ts ❌
- Don't use timestamps for ordering (clock skew issues)
- Don't compare all keys for sync (use Merkle trees)
- Don't block on all replicas (reduces availability)
- Don't ignore conflicts (vector clocks detect them)
- Don't skip compaction (WAL grows unbounded)
- Don't forget to test network partitions

### Key Takeaways
1. **CAP theorem**: Can't have all three (Consistency, Availability, Partition-tolerance)
2. **Quorum math**: R + W > N guarantees consistency
3. **Vector clocks**: Track causality, detect conflicts
4. **Merkle trees**: O(log N) sync vs O(N) comparison
5. **WAL**: Sequential writes for durability with low latency

This system demonstrates production-grade distributed systems design used by Amazon DynamoDB, Apache Cassandra, and Riak.
