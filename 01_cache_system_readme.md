# Advanced Cache System - Interview Guide

## 📋 Overview
A comprehensive cache system implementing LRU, LFU, and FIFO eviction policies with TTL (Time-to-Live) support and thread safety.

## 🎯 Interview Focus Areas

### Core Concepts to Master
1. **Cache Eviction Policies**: LRU, LFU, FIFO algorithms
2. **TTL Implementation**: Time-based cache expiration
3. **Thread Safety**: Concurrent access handling
4. **Design Patterns**: Strategy, Factory patterns
5. **Data Structures**: HashMap + Doubly Linked List (LRU), Frequency Maps (LFU)

## 🔥 Step-by-Step Implementation Guide

### Phase 1: Requirements Clarification (2-3 minutes)
**Always ask these questions:**
```
Q: What's the expected capacity of the cache?
Q: Do we need thread safety?
Q: Should we support TTL (expiration)?
Q: Which eviction policies are needed?
Q: Do we need to track cache statistics?
```

### Phase 2: High-Level Design (3-4 minutes)
1. **Draw the architecture**:
   ```
   CacheInterface
   ├── LRUCache
   ├── LFUCache
   └── FIFOCache
   
   CacheFactory → Creates appropriate cache
   ```

2. **Explain key components**:
   - Abstract Cache base class
   - Concrete implementations for each policy
   - CacheEntry with TTL metadata
   - Thread-safe operations using RLock

### Phase 3: Implementation (15-20 minutes)

#### Start with LRU (Most Common)
```python
# Key insight: HashMap + Doubly Linked List
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        # Sentinel nodes to avoid edge cases
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
```

**🎯 Interview Tip**: Always mention time complexity as you code:
- "This operation is O(1) because we use HashMap for lookup and maintain pointers"

#### Key Methods to Implement:
1. **get()**: Move to front, check TTL
2. **put()**: Add to front, evict from tail if needed
3. **Helper methods**: _add_to_front(), _remove_node(), _move_to_front()

## 📚 Critical Knowledge Points

### 1. LRU Implementation Details
```python
# Why doubly linked list?
# - O(1) insertion/deletion at any position
# - Easy to move nodes to front (mark as recently used)
# - Sentinel nodes eliminate edge case checks

def _move_to_front(self, node):
    self._remove_node(node)  # O(1) with pointers
    self._add_to_front(node)  # O(1) insertion
```

### 2. LFU Implementation Details
```python
# Key insight: Track frequency + use LRU within same frequency
class LFUCache:
    def __init__(self, capacity):
        self.cache = {}  # key -> (entry, frequency)
        self.freq_map = defaultdict(OrderedDict)  # freq -> {key: entry}
        self.min_freq = 0  # Track minimum frequency for eviction
```

### 3. TTL Implementation
```python
@dataclass
class CacheEntry:
    value: Any
    expiry_time: Optional[float] = None  # Unix timestamp
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        if self.expiry_time is None:
            return False
        return time.time() >= self.expiry_time
```

### 4. Thread Safety
```python
# Use RLock (Reentrant Lock) for nested method calls
from threading import RLock

def get(self, key):
    with self.lock:  # Ensures atomic operations
        entry = self._get_internal(key)
        if entry and entry.is_expired():
            self._delete_internal(key)
            return None
        return entry.value if entry else None
```

## ⚡ Performance Analysis

| Operation | LRU | LFU | FIFO |
|-----------|-----|-----|------|
| get() | O(1) | O(1) | O(1) |
| put() | O(1) | O(1) | O(1) |
| Space | O(n) | O(n) | O(n) |

## 🎯 Do's and Don'ts

### ✅ DO's
1. **Start with requirements**: Always clarify before coding
2. **Use appropriate data structures**: HashMap for O(1) access, LinkedList for order
3. **Handle edge cases**: Empty cache, capacity 0, null values
4. **Explain time complexity**: Mention Big-O as you implement
5. **Add thread safety**: Use locks for concurrent access
6. **Test your implementation**: Walk through examples

### ❌ DON'Ts
1. **Don't jump to coding**: Always discuss design first
2. **Don't ignore thread safety**: Mention even if you don't implement
3. **Don't hardcode values**: Use configuration objects
4. **Don't forget TTL checks**: Always validate expiration
5. **Don't use inefficient structures**: Avoid O(n) operations in hot paths
6. **Don't skip validation**: Check for null/invalid inputs

## 🎤 Expected Interview Questions & Answers

### Q1: "Why use doubly linked list for LRU instead of single linked list?"
**A**: "Doubly linked list allows O(1) deletion from middle of the list. With single linked list, we'd need O(n) to find the previous node for deletion. Since we need to move accessed nodes to front and remove LRU nodes, bidirectional pointers are essential for O(1) performance."

### Q2: "How would you handle cache stampede (multiple threads requesting same missing key)?"
**A**: "I'd implement a loading cache pattern:
1. Use a separate 'loading' set to track keys being fetched
2. First thread adds key to loading set and fetches data
3. Other threads wait or return stale data if available
4. Remove from loading set once fetch completes"

### Q3: "How would you implement cache warming?"
**A**: "Several approaches:
1. **Background refresh**: Async job refreshes popular keys before expiry
2. **Probability-based**: Refresh with probability inversely related to TTL remaining
3. **Write-behind**: Update cache on writes, sync to storage later
4. **Preload on startup**: Load frequently accessed data on application start"

### Q4: "What are the trade-offs between LRU vs LFU?"
**A**: 
- **LRU**: Better for temporal locality (recent access predicts future access). Simple, works well for most use cases.
- **LFU**: Better when access frequency matters more than recency. Good for scenarios with distinct popular vs unpopular items.
- **Trade-off**: LFU uses more memory (frequency counters) and can be slower to adapt to changing patterns."

### Q5: "How would you make this cache distributed?"
**A**: "Several approaches:
1. **Consistent hashing**: Distribute keys across nodes, handle node failures
2. **Replication**: Master-slave setup with async replication
3. **Cache-aside pattern**: Application manages cache, falls back to DB
4. **Write-through/Write-behind**: Different consistency guarantees
5. **Use Redis/Hazelcast**: Leverage existing distributed cache solutions"

### Q6: "How do you handle memory pressure?"
**A**: "Multiple strategies:
1. **Adaptive capacity**: Reduce cache size based on memory usage
2. **Priority eviction**: Evict low-priority items first regardless of policy
3. **Compression**: Compress cache values to save space
4. **Off-heap storage**: Use disk-backed cache for overflow
5. **Memory monitoring**: Track heap usage and trigger cleanup"

### Q7: "What about cache coherence in multi-level cache?"
**A**: "Common patterns:
1. **Write-through**: Updates propagate to all levels immediately
2. **Write-back**: Updates go to one level, sync later (eventual consistency)
3. **Invalidation-based**: Invalidate cache entries on updates
4. **Version-based**: Use version numbers to detect stale data
5. **Event-driven**: Use messaging to notify cache invalidations"

## 🧪 Testing Strategy

### Unit Tests to Write
```python
def test_lru_basic_operations():
    # Test get/put basic functionality
    
def test_lru_eviction():
    # Test LRU eviction when capacity exceeded
    
def test_ttl_expiration():
    # Test TTL-based expiration
    
def test_thread_safety():
    # Test concurrent access
    
def test_edge_cases():
    # Test capacity 0, null values, etc.
```

### Demo Scenarios
1. **Basic operations**: get/put/delete
2. **Eviction behavior**: Fill cache, observe which items get evicted
3. **TTL functionality**: Set items with different TTLs, observe expiration
4. **Thread safety**: Concurrent operations from multiple threads
5. **Performance**: Measure operations per second

## 🚀 Production Considerations

### Monitoring & Metrics
- Hit/miss ratios
- Eviction rates
- Memory usage
- Request latency
- Thread contention

### Configuration
- Dynamic capacity adjustment
- TTL policies per key pattern
- Eviction policy selection
- Background cleanup frequency

### Optimization
- Batch operations for better throughput
- Memory-mapped files for persistence
- Lock-free data structures for high concurrency
- Bloom filters for negative cache

## 📖 Additional Learning Resources

### Books
- "Designing Data-Intensive Applications" - Martin Kleppmann
- "High Performance Browser Networking" - Ilya Grigorik

### Papers
- "The LRU-K Page Replacement Algorithm" - O'Neil et al.
- "Caffeine: A high performance caching library" - Ben Manes

### Systems to Study
- Redis implementation
- Caffeine (Java high-performance cache)
- Guava Cache
- Hazelcast IMDG

---

## 💡 Final Interview Tips

1. **Start simple, then optimize**: Begin with basic implementation, add features incrementally
2. **Communicate trade-offs**: Always discuss alternatives and their pros/cons  
3. **Think about scale**: Mention how design changes with millions of users
4. **Consider failure modes**: What happens when things go wrong?
5. **Practice coding**: Be comfortable implementing LRU from memory
6. **Know the numbers**: Cache hit ratios, typical TTL values, memory constraints

Remember: The goal is to demonstrate your problem-solving approach, not just to produce working code. Explain your thinking process throughout!