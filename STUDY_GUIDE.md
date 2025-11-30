# 🎯 21-Day LLD Interview Bootcamp

<div align="center">

**Battle-Tested Study Plan for FAANG/Microsoft LLD Rounds**

*Transform from nervous candidate to confident system designer in 3 weeks*

**Daily Focus • Interview Tactics • Real Rebuttals • Zero Fluff**

</div>

---

## 📋 What Makes This Guide Different

❌ **NOT**: Generic advice, vague timelines, "study these topics"  
✅ **YES**: Exact daily tasks, real interview traps, specific comeback lines, timing strategies

**This guide is for you if:**
- You have an LLD interview in 3 weeks
- You've read theory but freeze in interviews
- You know patterns but can't apply them under pressure
- You need concrete daily actions, not motivational speeches

**Time Investment:** 3-4 hours/day (63-84 hours total)  
**Outcome:** Confidently solve 90% of LLD problems in 45-minute interviews

---

## 🎯 The 3-Week Strategy

### Week 1: Foundation + Core Systems (Survive)
Master the 5 systems that appear in 70% of interviews. Build muscle memory.

### Week 2: Advanced Patterns + Edge Cases (Compete)
Handle concurrency, scalability, and the tricky follow-ups that separate levels.

### Week 3: Mock Interviews + Rebuttals (Dominate)
Practice under pressure, learn to handle pushback, polish your presentation.

---

# 📅 WEEK 1: FOUNDATION & CORE SYSTEMS

## Day 1 (Monday): OOP + Cache System - The Foundation

**Goal:** Master the #1 most asked system and refresh OOP fundamentals


**🎯 Morning Session (90 mins): Theory**

1. **OOP Principles Refresh (30 mins)**
   - Encapsulation, Inheritance, Polymorphism, Abstraction
   - **Critical Interview Question:** "Why composition over inheritance?"
   - **Your Answer:** "Inheritance creates tight coupling. Composition allows runtime behavior changes and better testability. Example: CacheStrategy interface vs extending AbstractCache."

2. **SOLID Principles (30 mins)**
   - Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
   - **Memorize This:** "SRP = one class, one reason to change. OCP = extend via interfaces, not modifications."
   - **Interview Trap:** Interviewer says "Just make everything public for simplicity" 
   - **Your Rebuttal:** "In production, that violates encapsulation. I'll use private fields with getters/setters for thread-safety and validation."

3. **Study Implementation: `01_cache_system.py` (30 mins)**
   - Read the entire implementation
   - Understand: LRU eviction, capacity management, O(1) operations
   - **Key Pattern:** HashMap + Doubly Linked List

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

4. **Code From Scratch - LRU Cache (60 mins)**
   - Close the file, implement without looking
   - Must have: `get()`, `put()`, `_evict()`, capacity check
   - Test with: empty cache, full cache, repeated keys
   - **Time yourself:** Should complete basic version in 30 mins

5. **Handle Follow-ups (30 mins)**
   - "Make it thread-safe" → Add `threading.Lock()` around operations
   - "Support TTL expiration" → Add `expiry_time` field, cleanup thread
   - "What if memory limit, not item count?" → Track `total_bytes`, evict accordingly

**🎯 Evening Session (45 mins): Interview Prep**

6. **Memorize Your Opening (15 mins)**
   ```
   "I'll design an LRU cache with O(1) get and put operations.
   Core components: HashMap for O(1) lookup, Doubly Linked List for O(1) eviction.
   I'll start with the data structures, then implement operations, then handle edge cases.
   Any specific requirements on thread-safety or capacity?"
   ```

7. **Practice Explaining (30 mins)**
   - Set a timer for 5 minutes
   - Explain your design to a rubber duck/friend/recording
   - **Must cover:** Data structures choice, time complexity, trade-offs
   - **Common mistake:** Jumping to code without explaining approach

**✅ Day 1 Success Criteria:**
- [ ] Can implement LRU cache in 30 mins without reference
- [ ] Can explain O(1) operations for get/put
- [ ] Can handle 3+ follow-up variations
- [ ] Memorized opening framework for any problem

---

## Day 2 (Tuesday): Rate Limiter - Concurrency Basics

**Goal:** Master concurrency patterns and the #2 most asked system

**🎯 Morning Session (90 mins): Concurrency Fundamentals**

1. **Threading Concepts (45 mins)**
   - Race conditions, deadlocks, thread-safety
   - **Critical for Interviews:** Understanding `Lock()`, `Semaphore()`, `Queue()`
   - **Study:** `02_rate_limiter.py` - Token Bucket implementation
   
2. **Algorithm Deep Dive (45 mins)**
   - Token Bucket vs Sliding Window vs Fixed Window
   - **Interview Question:** "Which algorithm and why?"
   - **Your Answer:** "Token Bucket for smooth rate limiting, allows bursts. Sliding Window for strict per-second limits. I'll implement Token Bucket because it's more forgiving for user experience."
   
   - **Trap:** Interviewer: "Why not just count requests?"
   - **Rebuttal:** "Simple counter has edge case: 100 requests at 00:00:59, 100 at 00:01:01 = 200 requests in 2 seconds but passes per-minute check."

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

3. **Build Rate Limiter (60 mins)**
   - Implement Token Bucket from scratch
   - Components: `tokens`, `last_refill_time`, `allow_request()`
   - Add thread-safety with `threading.Lock()`
   - **Test:** Multiple threads making concurrent requests

4. **Distributed Extension (30 mins)**
   - "How would you scale this?"
   - **Answer Structure:**
   ```
   1. Problem: In-memory state doesn't work across servers
   2. Solution: Use Redis with INCR and EXPIRE
   3. Implementation: Redis Lua script for atomicity
   4. Trade-off: Network latency vs accuracy
   ```

**🎯 Evening Session (45 mins): Interview Scenarios**

5. **Practice Rebuttals (45 mins)**
   
   **Scenario 1:** "Your Token Bucket allows bursts. That's a security risk."
   **Your Response:** "Valid concern. I can add a secondary sliding window check to cap bursts, or implement a Leaky Bucket variant that enforces steady rate. Which aligns better with requirements?"

   **Scenario 2:** "This is too complex. Why not just count?"
   **Your Response:** "Simple counting has timing edge cases [explain]. For production, Token Bucket is industry standard (used by AWS, Stripe). The complexity is justified by correctness."

   **Scenario 3:** "How do you handle clock skew across servers?"
   **Your Response:** "Use Redis TIME command for server-side timestamps, not client time. For stricter guarantees, implement logical clocks or centralized rate limiter service."

**✅ Day 2 Success Criteria:**
- [ ] Can implement Token Bucket in 25 mins
- [ ] Can explain thread-safety approach
- [ ] Can compare 3 algorithms with trade-offs
- [ ] Have 3 prepared rebuttals for pushback

---

## Day 3 (Wednesday): Parking Lot - OOP Design Excellence

**Goal:** Master class design, relationships, and design patterns

**🎯 Morning Session (90 mins): Design Patterns**

1. **Essential Patterns (60 mins)**
   - **Singleton:** ParkingLot instance (thread-safe initialization)
   - **Factory:** VehicleFactory for Car/Truck/Motorcycle
   - **Strategy:** PricingStrategy interface (hourly/daily/flat)
   - **Observer:** NotificationService for spot availability

2. **Class Diagram Practice (30 mins)**
   - Draw: ParkingLot, Floor, Spot, Vehicle, Ticket classes
   - **Interview Tip:** Always start with nouns (classes) and verbs (methods)
   - **Relationships:** Composition (ParkingLot HAS Floors), Inheritance (Vehicle types)

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

3. **Code Parking Lot System (70 mins)**
   - Start from `07_parking_lot.py` structure
   - Implement: `park_vehicle()`, `unpark_vehicle()`, `get_available_spots()`
   - **Key Method:** `find_available_spot()` - use size-based filtering
   
4. **Design Decisions (20 mins)**
   - "How to find nearest available spot?" → Floor-by-floor search with spot indexing
   - "How to handle payment failures?" → Reservation timeout, spot unlocking
   - "Database schema?" → spots, tickets, vehicles tables with foreign keys

**🎯 Evening Session (60 mins): Interview Communication**

5. **Clarifying Questions Template (30 mins)**
   
   **Always Ask First:**
   ```
   1. Scale: How many floors/spots? Affects data structures
   2. Vehicle types: Just cars or trucks/motorcycles? Affects spot sizing
   3. Features: Payment, reservations, handicap spots? Affects complexity
   4. Concurrent access: Multiple entry points? Affects locking strategy
   ```

6. **Class Presentation Practice (30 mins)**
   - Present your design in 10 minutes
   - **Structure:**
     1. High-level components (30 seconds)
     2. Core classes and relationships (3 mins)
     3. Key methods with logic (4 mins)
     4. Edge cases handled (2 mins)
     5. Open for questions (30 seconds)

**✅ Day 3 Success Criteria:**
- [ ] Can design class diagram in 10 mins
- [ ] Can identify 4+ design patterns in any system
- [ ] Can present design clearly in 10 mins
- [ ] Have 5 clarifying questions memorized

---

## Day 4 (Thursday): URL Shortener - Scalability Thinking

**Goal:** Learn to think about distributed systems and data modeling

**🎯 Morning Session (90 mins): System Thinking**

1. **Study Implementation (45 mins)**
   - Analyze `08_url_shortener.py`
   - **Focus:** ID generation (Base62 encoding), collision handling, analytics

2. **Scalability Deep Dive (45 mins)**
   - **Key Question:** "How do you generate unique IDs?"
   - **Evolution of Answers:**
     - ❌ Weak: "Random strings"
     - ⚠️ OK: "Hash the URL"
     - ✅ Strong: "Counter-based with Base62, or Twitter Snowflake IDs"
     - 🏆 Senior: "Pre-generate ID ranges per server, use ZooKeeper for coordination"

   - **Follow-up:** "What if 2 users shorten same URL?"
   - **Your Decision Framework:** "Depends on requirements. Same short URL = save space but privacy concern. Different short URLs = better analytics and user isolation."

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation + DB Design**

3. **Implement Core System (50 mins)**
   - Build: `shorten_url()`, `expand_url()`, `get_analytics()`
   - **Must Have:** Collision handling, expiration logic, analytics tracking

4. **Database Schema (40 mins)**
   ```sql
   urls:
     - short_code (PK, indexed)
     - long_url (TEXT)
     - user_id (FK, indexed)
     - created_at, expires_at
     - click_count
   
   analytics:
     - short_code (FK)
     - timestamp
     - ip_address
     - user_agent
     - referer
   ```
   
   - **Interview Trap:** "Why separate analytics table?"
   - **Rebuttal:** "High write volume for clicks. Separate table prevents lock contention on main urls table. Can shard analytics independently."

**🎯 Evening Session (45 mins): Advanced Scenarios**

5. **Handle Tough Questions (45 mins)**

   **Q: "Your Base62 encoding is sequential. Attackers can enumerate all URLs."**
   **A:** "Good catch. Solutions: 1) Add random salt to counter before encoding, 2) Use cryptographic hash of counter + secret, 3) Shuffle Base62 alphabet. I'll use salted counter - maintains uniqueness, adds randomness."

   **Q: "How do you handle 1000 requests/sec for same short URL?"**
   **A:** "Caching strategy: CDN for redirect responses, Redis for hot URLs (99% cache hit), database with read replicas. Redirect response has HTTP 301 (permanent) - browsers cache it."

   **Q: "User wants custom short codes like 'google.com/promo'"**
   **A:** "Maintain reserved keywords list, validate custom code doesn't conflict with generated codes. Store custom codes in separate table column or use prefix differentiation."

**✅ Day 4 Success Criteria:**
- [ ] Can design scalable ID generation strategy
- [ ] Can defend design decisions with trade-offs
- [ ] Can propose DB schema with indexing strategy
- [ ] Can handle 3+ advanced scalability questions

---

## Day 5 (Friday): Elevator System - State Machines

**Goal:** Master state management and scheduling algorithms

**🎯 Morning Session (90 mins): Algorithm Study**

1. **Elevator Algorithms (60 mins)**
   - **FCFS:** First Come First Served (simple, inefficient)
   - **SCAN:** Sweep up then down (elevator direction)
   - **LOOK:** Like SCAN but reverses at last request
   - **Destination Dispatch:** Pre-assign elevators (modern systems)
   
   - **Interview Question:** "Which algorithm?"
   - **Your Answer:** "LOOK for single elevator - efficient and intuitive. For multiple elevators, nearest car assignment with direction consideration."

2. **State Machine Design (30 mins)**
   - States: IDLE, MOVING_UP, MOVING_DOWN, DOOR_OPEN, MAINTENANCE
   - Transitions: User request triggers state changes
   - **Pattern:** State Pattern for clean transitions

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

3. **Build Elevator Controller (75 mins)**
   - Study `04_elevator_system.py`
   - Implement request queue (up requests, down requests)
   - **Core Logic:** 
     ```python
     def process_requests(self):
         if moving_up:
             handle_up_requests_on_path()
             if no_more_up:
                 reverse_to_down()
         # Similar for down
     ```

4. **Edge Cases (15 mins)**
   - Emergency stop request
   - Weight overload detection
   - Power failure recovery
   - Concurrent requests from multiple floors

**🎯 Evening Session (45 mins): Week 1 Review**

5. **Systems Review (30 mins)**
   - LRU Cache: HashMap + DLL, O(1) operations
   - Rate Limiter: Token Bucket, thread-safety
   - Parking Lot: OOP design, patterns
   - URL Shortener: Scalability, ID generation
   - Elevator: State machines, scheduling

6. **Self-Assessment (15 mins)**
   - Can you implement each system in 30 mins? If not, practice this weekend.
   - Can you explain trade-offs for each design decision?
   - Can you handle "why not X?" questions confidently?

**✅ Day 5 Success Criteria:**
- [ ] Understand LOOK algorithm for elevator scheduling
- [ ] Can implement state machine pattern
- [ ] Can explain 5 systems from Week 1 confidently
- [ ] Ready for Week 2 advanced topics

---

## Weekend (Sat-Sun): Consolidation & Practice

**🎯 Saturday (3 hours)**

1. **Speed Practice (90 mins)**
   - Set timer: 30 minutes per system
   - Implement LRU Cache, Rate Limiter, URL Shortener from memory
   - **Goal:** Build muscle memory, reduce thinking time

2. **Weak Areas (90 mins)**
   - Which system took longest? Re-study it
   - Which concepts are unclear? Review those sections
   - Practice explaining your weakest system out loud

**🎯 Sunday (2 hours)**

1. **Mock Interview (60 mins)**
   - Use `questions.txt` to pick a random problem
   - Set 45-minute timer
   - Talk through entire process: clarify → design → implement → test
   - Record yourself if possible

2. **Reflection (60 mins)**
   - What went well? What needs work?
   - Update your clarifying questions template
   - Note common mistakes you make

**✅ Weekend Goal:**
- [ ] Can implement Week 1 systems in 30 mins each
- [ ] Completed at least 1 full mock interview
- [ ] Identified top 3 areas to improve

---

# 📅 WEEK 2: ADVANCED PATTERNS & CONCURRENCY

## Day 6 (Monday): Job Processor - Producer-Consumer Pattern

**Goal:** Master multithreading and queue-based architectures

**🎯 Morning Session (90 mins): Concurrency Deep Dive**

1. **Producer-Consumer Pattern (45 mins)**
   - **Core Concept:** Decoupling producers from consumers with queue
   - `Queue.put()` - blocking when full
   - `Queue.get()` - blocking when empty
   - Thread pool for workers
   
2. **Study Implementation (45 mins)**
   - Analyze `03_job_processor.py`
   - **Key Components:** JobQueue, WorkerPool, PriorityQueue for priorities
   - **Pattern:** Observer for status updates, Strategy for retry logic

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Build & Extend**

3. **Implement Job Processor (60 mins)**
   - Core: `submit_job()`, `worker_thread()`, `process_job()`
   - Add: Priority levels, retry with exponential backoff
   - Handle: Worker death, job timeout, graceful shutdown

4. **Advanced Features (30 mins)**
   - **Dead Letter Queue:** Failed jobs after max retries
   - **Job Dependencies:** Job B runs after Job A completes
   - **Rate Limiting:** Max concurrent jobs per type

**🎯 Evening Session (45 mins): Interview Scenarios**

5. **Tough Questions (45 mins)**

   **Q: "How do you prevent thread pool exhaustion?"**
   **A:** "1) Queue size limits - reject or block new submissions when full, 2) Job timeouts - kill hung jobs, 3) Priority-based scheduling - critical jobs first, 4) Monitoring and alerting for queue depth."

   **Q: "What if a job modifies shared state unsafely?"**
   **A:** "Job isolation principle: each job should be idempotent and stateless. If shared state needed, use locks or message passing. Better design: jobs return results, coordinator updates state."

   **Q: "How do you handle 1 million jobs/second?"**
   **A:** "Vertical: increase threads, use async I/O. Horizontal: partition jobs by type, shard across workers. External: use proper message queue like RabbitMQ/Kafka with consumer groups."

**✅ Day 6 Success Criteria:**
- [ ] Can implement producer-consumer from scratch
- [ ] Understand thread safety for shared queue
- [ ] Can add priority and retry mechanisms
- [ ] Can discuss scaling to distributed systems

---

## Day 7 (Tuesday): Distributed Cache - Consistency & Replication

**Goal:** Learn distributed systems concepts and consistency models

**🎯 Morning Session (90 mins): Distributed Systems Fundamentals**

1. **Core Concepts (60 mins)**
   - **CAP Theorem:** Can't have all 3 (Consistency, Availability, Partition Tolerance)
   - **Consistency Models:**
     - Strong: All nodes see same data immediately (expensive)
     - Eventual: Nodes converge over time (performant)
     - Causal: Preserves order of related operations
   
   - **Interview Question:** "Strong or eventual consistency?"
   - **Your Framework:**
     ```
     1. Ask: "What's the read:write ratio and staleness tolerance?"
     2. For social media feeds: Eventual (users tolerate slight delays)
     3. For financial transactions: Strong (can't show wrong balance)
     4. I'll design for eventual with option to upgrade to strong
     ```

2. **Study Implementation (30 mins)**
   - Analyze `13_distributed_cache.py`
   - **Key Techniques:** Consistent hashing, replication factor, gossip protocol

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

3. **Build Distributed Cache (60 mins)**
   - **Consistent Hashing:** Virtual nodes for load balancing
   ```python
   def get_node(self, key):
       hash_value = hash(key)
       # Find next node clockwise on ring
       for node_hash in sorted(self.ring.keys()):
           if hash_value <= node_hash:
               return self.ring[node_hash]
       return self.ring[min(self.ring.keys())]
   ```
   
   - **Replication:** Write to N nodes, read from quorum
   - **Conflict Resolution:** Last-write-wins with vector clocks

4. **Failure Handling (30 mins)**
   - Node failure detection: Heartbeat mechanism
   - Data migration: Replica promotion
   - Network partition: Sloppy quorum (hinted handoff)

**🎯 Evening Session (45 mins): Senior-Level Questions**

5. **Advanced Scenarios (45 mins)**

   **Q: "Vector clocks can grow unbounded. How do you handle that?"**
   **A:** "Use dotted version vectors or version stamps with pruning. Alternative: use timestamps + conflict-free replicated data types (CRDTs) for certain data structures. Trade-off: timestamp-based has clock skew issues."

   **Q: "Your consistent hashing causes hotspots when nodes added/removed."**
   **A:** "Virtual nodes address this - each physical node appears multiple times on ring (100-200 virtual nodes). When adding node, load distributes evenly. Can also use jump consistent hash for better distribution."

   **Q: "How do you invalidate cache entries across all nodes?"**
   **A:** "Options: 1) Broadcast invalidation messages (expensive), 2) TTL-based expiration (may serve stale), 3) Versioning with lazy invalidation, 4) Centralized invalidation log with polling. Choose based on staleness tolerance."

**✅ Day 7 Success Criteria:**
- [ ] Can explain CAP theorem with examples
- [ ] Can implement consistent hashing
- [ ] Understand replication and quorum reads
- [ ] Can discuss trade-offs in consistency models

---

## Day 8 (Wednesday): Distributed KV Store - Sharding & Partitioning

**Goal:** Master data partitioning and fault tolerance

**🎯 Morning Session (90 mins): Partitioning Strategies**

1. **Sharding Techniques (60 mins)**
   - **Range-based:** Keys A-M on shard1, N-Z on shard2 (uneven load)
   - **Hash-based:** hash(key) % N (rebalancing needed on resize)
   - **Consistent Hashing:** Minimal data movement on topology changes (best)
   
   - **Interview Question:** "How do you handle hotspots?"
   - **Your Answer:** "Identify hot keys via monitoring, use sub-sharding for specific keys, or add read replicas. For celebrity user scenario, can cache hot keys in separate high-capacity tier."

2. **Study Implementation (30 mins)**
   - Analyze `14_distributed_kv_store.py`
   - Focus on: Partition assignment, read/write quorum, conflict resolution

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Fault Tolerance**

3. **Implement Replication Logic (60 mins)**
   ```python
   def put(self, key, value):
       # Write to N replicas
       nodes = self.get_replica_nodes(key, replication_factor=3)
       successful_writes = 0
       for node in nodes:
           if node.write(key, value):
               successful_writes += 1
       # Quorum: W + R > N ensures consistency
       if successful_writes >= self.write_quorum:  # W=2
           return True
       return False  # Failed to meet quorum
   ```

4. **Failure Scenarios (30 mins)**
   - **Node crash during write:** Use write-ahead log for recovery
   - **Network partition:** Accept writes on majority partition only
   - **Data corruption:** Checksums + merkle trees for validation
   - **Split brain:** Use ZooKeeper or etcd for coordination

**🎯 Evening Session (45 mins): System Design Presentation**

5. **Present Complete Design (45 mins)**
   - Practice explaining your KV store in 15 minutes
   - **Structure:**
     1. "I'll design a distributed KV store with high availability and eventual consistency"
     2. "Core components: partitioning via consistent hashing, replication factor 3, quorum-based reads/writes"
     3. "Write path: hash key → find replicas → write to N nodes → return on W successes"
     4. "Read path: read from R nodes → resolve conflicts with vector clocks → return latest"
     5. "Failure handling: heartbeat monitoring, anti-entropy repair, hinted handoff"

**✅ Day 8 Success Criteria:**
- [ ] Can explain 3 partitioning strategies with trade-offs
- [ ] Can implement quorum-based read/write
- [ ] Can handle failure scenarios confidently
- [ ] Can present complete distributed system design

---

## Day 9 (Thursday): File Storage System - Chunking & Sync

**Goal:** Master file handling, versioning, and sync algorithms

**🎯 Morning Session (90 mins): File System Design**

1. **Chunking Strategy (45 mins)**
   - **Why Chunk?** Large files, incremental uploads, resume capability, deduplication
   - **Fixed-size chunks:** Simple, predictable (Dropbox uses 4MB)
   - **Content-defined chunks:** Better deduplication (variable size based on content)
   
   - **Interview Question:** "How do you handle concurrent edits?"
   - **Your Answer:** "Operational Transform or CRDT for real-time collab. For file-level: last-write-wins with conflict copies, or 3-way merge. I'll implement version-based conflict detection."

2. **Study Implementation (45 mins)**
   - Analyze `16_file_storage_system.py`
   - **Key Components:** ChunkManager, VersionControl, SyncEngine, MetadataStore

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Implementation**

3. **Build Core Features (70 mins)**
   
   **File Upload:**
   ```python
   def upload_file(self, file_path):
       chunks = self.chunk_file(file_path, chunk_size=4*1024*1024)
       chunk_hashes = []
       for chunk in chunks:
           chunk_hash = hashlib.sha256(chunk).hexdigest()
           if not self.chunk_exists(chunk_hash):  # Deduplication
               self.store_chunk(chunk_hash, chunk)
           chunk_hashes.append(chunk_hash)
       version_id = self.create_version(file_path, chunk_hashes)
       return version_id
   ```

   **Sync Algorithm:**
   ```python
   def sync(self, client_file_list):
       server_files = self.get_file_metadata()
       # Compare versions using Merkle tree or timestamp
       to_upload = [f for f in client_file_list if needs_upload(f)]
       to_download = [f for f in server_files if needs_download(f)]
       return {'upload': to_upload, 'download': to_download}
   ```

4. **Advanced Features (20 mins)**
   - **Versioning:** Store metadata with parent version IDs
   - **Sharing:** Access control lists (ACLs) with permissions
   - **Conflict Resolution:** Create conflict copy with timestamp

**🎯 Evening Session (45 mins): Security & Performance**

5. **Critical Considerations (45 mins)**

   **Q: "How do you secure files?"**
   **A:** "Multi-layer: 1) TLS for transit, 2) AES-256 encryption at rest with user-derived keys, 3) Encrypt chunk hashes in metadata, 4) Zero-knowledge architecture - server can't decrypt user files."

   **Q: "How do you handle a 100GB file?"**
   **A:** "Chunking with resumable uploads, parallel chunk uploads (5-10 concurrent), streaming for memory efficiency. Use multipart upload API. Progress tracking via chunk completion bitmap."

   **Q: "User deletes file by accident. How to recover?"**
   **A:** "Soft delete with retention period (30 days), version history for 30 days, trash folder with restore capability. After retention, garbage collect chunks not referenced by any file."

**✅ Day 9 Success Criteria:**
- [ ] Can design chunking strategy with deduplication
- [ ] Can implement sync algorithm
- [ ] Can explain versioning and conflict resolution
- [ ] Can discuss security and performance optimizations

---

## Day 10 (Friday): Autocomplete - Trie & Ranking

**Goal:** Master tree structures and ranking algorithms

**🎯 Morning Session (90 mins): Trie Implementation**

1. **Trie Data Structure (60 mins)**
   ```python
   class TrieNode:
       def __init__(self):
           self.children = {}  # char -> TrieNode
           self.is_end = False
           self.frequency = 0  # For ranking
           self.top_suggestions = []  # Cache top 10
   
   def insert(self, word, frequency):
       node = self.root
       for char in word:
           if char not in node.children:
               node.children[char] = TrieNode()
           node = node.children[char]
       node.is_end = True
       node.frequency = frequency
   
   def search_prefix(self, prefix):
       node = self.root
       for char in prefix:
           if char not in node.children:
               return []
           node = node.children[char]
       # Collect all words from this node
       return self.collect_words(node, prefix)
   ```

2. **Ranking Strategies (30 mins)**
   - Frequency-based: Most searched terms first
   - Recency-weighted: Recent searches rank higher
   - Personalized: User history influences results
   - Typo-tolerant: Edit distance matching

**☕ Break (15 mins)**

**🎯 Afternoon Session (90 mins): Optimizations**

3. **Cache Top Suggestions (45 mins)**
   ```python
   # Pre-compute top 10 at each node
   def update_top_suggestions(self, node):
       all_words = self.collect_with_frequency(node)
       # Sort by frequency, take top 10
       node.top_suggestions = sorted(all_words, 
                                     key=lambda x: x[1], 
                                     reverse=True)[:10]
   ```
   
   - **Trade-off:** More memory, but O(1) autocomplete instead of O(N) tree traversal

4. **Handle Scale (45 mins)**
   - **Problem:** Trie doesn't fit in memory (billions of queries)
   - **Solutions:**
     1. Prefix sharding: "a*" on server1, "b*" on server2
     2. Compress trie with LOUDS encoding
     3. Cache popular prefixes in Redis
     4. Use distributed trie with consistent hashing

**🎯 Evening Session (45 mins): Week 2 Review**

5. **Systems Review (30 mins)**
   - Job Processor: Producer-consumer, thread pool
   - Distributed Cache: Consistent hashing, replication
   - Distributed KV Store: Partitioning, quorum
   - File Storage: Chunking, versioning, sync
   - Autocomplete: Trie, ranking, caching

6. **Pattern Recognition (15 mins)**
   - Identify common patterns across systems:
     - **Partitioning:** Consistent hashing appears in cache, KV store
     - **Replication:** Used for fault tolerance in distributed systems
     - **Caching:** Appears at multiple layers (client, server, CDN)
     - **Queueing:** Decouples components (job processor, messaging)

**✅ Day 10 Success Criteria:**
- [ ] Can implement Trie with autocomplete in 25 mins
- [ ] Can explain ranking and caching strategies
- [ ] Can discuss scaling to billions of queries
- [ ] Understand patterns across Week 2 systems

---

## Weekend 2 (Sat-Sun): Advanced Practice & Mock Interviews

**🎯 Saturday (4 hours)**

1. **System Combination Practice (2 hours)**
   - Design a system combining multiple concepts:
   - **Problem:** "Design a real-time collaborative document editor"
   - **Your approach:**
     - File Storage system for document chunks
     - Distributed cache for active documents
     - Job processor for async operations (PDF export)
     - Autocomplete for mentions and commands
   - Practice explaining how components interact

2. **Concurrency Deep Dive (2 hours)**
   - Review all threading code from Week 2
   - Practice explaining: locks, semaphores, atomic operations
   - Implement thread-safe data structures from scratch

**🎯 Sunday (4 hours)**

1. **Mock Interviews (3 hours)**
   - Do 3 full 45-minute mock interviews
   - Use problems from `advanced_questions_1.md`
   - **Suggested:**
     1. Concurrent LRU Cache (threading focus)
     2. Consistent Hashing Data Structure (distributed systems)
     3. WebSocket Chat Server (real-time systems)
   
   - Record yourself, review for:
     - Clarity of explanation
     - Speed of implementation
     - Handling of follow-ups
     - Confidence in rebuttals

2. **Reflection & Adjustment (1 hour)**
   - What patterns are you missing?
   - Which rebuttals need work?
   - Update your personal cheat sheet with learnings

**✅ Weekend 2 Goal:**
- [ ] Completed 3 full mock interviews
- [ ] Can combine multiple system concepts
- [ ] Identified remaining weak areas for Week 3

---

# 📅 WEEK 3: POLISH, MOCKS, & INTERVIEW TACTICS

## Day 11-12 (Mon-Tue): Remaining Systems Speed Run

**Goal:** Cover remaining systems quickly for breadth

**🎯 Each System (90 mins each)**

**Day 11 Morning: Vending Machine**
- State machine pattern (IDLE, SELECT, PAYMENT, DISPENSE)
- Inventory management, payment processing
- **Key Interview Point:** State transitions and error handling

**Day 11 Afternoon: Library Management**
- CRUD operations, search functionality
- Fine calculation, reservation system
- **Key Interview Point:** Database design and indexing

**Day 12 Morning: Chess Game**
- Board representation (2D array), piece movement validation
- Check/checkmate detection, special moves (castling, en passant)
- **Key Interview Point:** Polymorphism for piece types

**Day 12 Afternoon: Snake & Ladder**
- Board representation, player management
- Random dice rolls, win condition
- **Key Interview Point:** Game loop and turn management

**Day 12 Evening: Notification Service**
- Multiple channels (email, SMS, push)
- Template management, retry logic
- **Key Interview Point:** Strategy pattern and priority queuing

**✅ Days 11-12 Success Criteria:**
- [ ] Understand core logic of each system
- [ ] Can explain unique patterns in each
- [ ] Have breadth across all 16 systems

---

## Day 13 (Wednesday): Interview Communication Mastery

**Goal:** Perfect your interview communication style

**🎯 Morning Session (2 hours): The Opening Framework**

1. **Problem Clarification (30 mins)**
   - Practice your standard opening:
   ```
   "Let me clarify requirements before designing.
   1. Scale: Expected users/requests per second?
   2. Features: Core features vs nice-to-have?
   3. Constraints: Latency requirements, consistency needs?
   4. Context: Standalone system or integrating with existing?
   
   Based on answers, I'll design for [specific approach]."
   ```

2. **Design Presentation (45 mins)**
   - **The 5-Minute Framework:**
   ```
   1. High-Level (1 min):
      "I'll design [system] with [key components]. Main challenges are [X, Y, Z]."
   
   2. Core Components (2 mins):
      "Three main classes: [Class A] handles [responsibility],
       [Class B] manages [responsibility], [Class C] coordinates [responsibility]."
   
   3. Key Interactions (1.5 mins):
      "Typical flow: User calls method X, which triggers Y, resulting in Z.
       For edge case [scenario], we handle with [approach]."
   
   4. Trade-offs (30 seconds):
      "This design prioritizes [X] over [Y] because [reason].
       Alternative would be [Z] but that has [drawback]."
   ```

3. **Live Coding Narration (45 mins)**
   - Practice coding while talking:
   - **Good:** "I'll create a HashMap to store key-value pairs for O(1) lookup..."
   - **Bad:** *Silence while typing*
   - **Good:** "This edge case needs handling - what if capacity is zero?"
   - **Bad:** "Hmm... let me think..." *long pause*

**☕ Break (15 mins)**

**🎯 Afternoon Session (2 hours): Rebuttal Mastery**

4. **Common Pushback & Responses (90 mins)**

   **Category 1: Oversimplification Accusation**
   
   **Interviewer:** "This seems too simple. You're missing [complex thing]."
   **Your Response:** "You're right that production systems need [complex thing]. I started with core functionality to validate approach. Let me extend to include [complex thing] - here's how..." *Proceeds to add complexity*
   **Why it works:** Shows you can think in layers, aren't defensive

   **Category 2: Performance Challenges**
   
   **Interviewer:** "This won't scale to 1 million requests/second."
   **Your Response:** "Agreed, single-node design has limits. For that scale, I'd: 1) Partition across N nodes using consistent hashing, 2) Add caching layer, 3) Use async processing. Which area would you like me to detail?"
   **Why it works:** Shows you know scaling techniques, asks for direction

   **Category 3: Alternative Suggestions**
   
   **Interviewer:** "Why not use [different approach]?"
   **Your Response:** "That's a valid alternative. Trade-offs: [Your approach] gives [benefit X] but [drawback Y], [Their approach] gives [benefit A] but [drawback B]. Given requirement [Z], I chose mine, but happy to pivot if [condition changes]."
   **Why it works:** Shows you evaluate trade-offs, not married to one solution

   **Category 4: Edge Case Testing**
   
   **Interviewer:** "What if [weird edge case]?"
   **Your Response:** "Good catch. Current design handles with [existing mechanism] OR needs addition of [new mechanism]. Let me add validation here..." *Updates code*
   **Why it works:** Shows you test mental models, iterate gracefully

   **Category 5: Time Pressure**
   
   **Interviewer:** "We only have 10 minutes left."
   **Your Response:** "I'll prioritize. Core implementation is complete. I can either: 1) Add error handling, 2) Optimize this method, 3) Discuss scaling. What's most valuable?"
   **Why it works:** Shows priority management, asks what matters

5. **Difficult Interviewer Types (30 mins)**

   **The Silent Interviewer:** Gives no feedback
   - **Strategy:** Narrate your thought process more, ask explicit check-ins: "Does this approach seem reasonable so far?"

   **The Nitpicker:** Focuses on syntax/minor details
   - **Strategy:** "I can fix syntax after logic is correct - would you like me to focus on correctness first?" Then perfect it at end.

   **The Skeptic:** Doubts every decision
   - **Strategy:** Stay calm, use data: "This is industry standard (used by X company). Alternative would be Y, but benchmark shows..." Have confidence in your choices.

   **The Scope Creeper:** Keeps adding requirements
   - **Strategy:** "I can add that feature. Should I extend current design or would you like me to restart with new requirements?"

**🎯 Evening Session (90 mins): Video Mock Interview**

6. **Record Full Interview (45 mins)**
   - Pick a problem you haven't done
   - Set up camera, do full 45-minute interview
   - Talk through everything as if interviewer present

7. **Self-Review (45 mins)**
   - Watch yourself, note:
     - Filler words (um, uh, like)
     - Long pauses without explanation
     - Unclear explanations
     - Weak voice tonality
   - **Action items:** What to fix tomorrow

**✅ Day 13 Success Criteria:**
- [ ] Have memorized opening framework
- [ ] Can handle 5 types of pushback confidently
- [ ] Completed full recorded mock interview
- [ ] Identified communication improvements needed

---

## Day 14 (Thursday): Advanced Scenarios & Edge Cases

**Goal:** Master the follow-up questions that differentiate levels

**🎯 Morning Session (2 hours): Advanced Extensions**

1. **Security Deep Dive (60 mins)**
   
   **Common Question:** "How do you prevent [attack]?"
   
   - **SQL Injection:** Prepared statements, input validation
   - **DDoS:** Rate limiting, CAPTCHA, CDN with DDoS protection
   - **Data Breach:** Encryption at rest/transit, key management, audit logs
   - **Unauthorized Access:** Authentication (JWT), authorization (RBAC), session management
   
   **Practice Response:**
   "For production security, I'd implement defense in depth:
   1. Input validation - whitelist known patterns
   2. Authentication - OAuth 2.0 with JWT tokens
   3. Authorization - role-based access control
   4. Encryption - AES-256 for data at rest, TLS 1.3 for transit
   5. Audit logging - track all sensitive operations
   6. Rate limiting - prevent abuse
   Which area should I detail?"

2. **Monitoring & Observability (60 mins)**
   
   **Common Question:** "How do you monitor this system?"
   
   **Your Framework:**
   ```
   Metrics (RED method):
   - Rate: requests per second
   - Errors: error rate, types
   - Duration: latency percentiles (p50, p95, p99)
   
   Logging:
   - Structured logs with correlation IDs
   - Error logs with stack traces
   - Audit logs for security events
   
   Tracing:
   - Distributed tracing for request flow
   - Span instrumentation for key operations
   
   Alerting:
   - Error rate > 1%: Page on-call
   - Latency p99 > 1s: Warning
   - Queue depth > 1000: Alert
   ```

**☕ Break (15 mins)**

**🎯 Afternoon Session (2 hours): Senior-Level Trade-offs**

3. **Consistency vs Performance (45 mins)**
   
   **Scenario:** "Your system is slow. Speed it up."
   
   **Your Analysis Process:**
   ```
   1. Identify bottleneck:
      "Let me profile. Is it CPU, I/O, network, or lock contention?"
   
   2. Propose solutions with trade-offs:
      - Add caching: +Speed, -Consistency (stale data)
      - Async processing: +Throughput, -Latency (initial request)
      - Denormalize data: +Read speed, -Write complexity
      - Add read replicas: +Read throughput, -Write contention
   
   3. Ask about constraints:
      "What's acceptable latency? Can we tolerate eventual consistency?"
   
   4. Implement chosen solution:
      *Show code changes*
   ```

4. **Testing Strategies (45 mins)**
   
   **Common Question:** "How would you test this?"
   
   **Your Response:**
   ```
   Unit Tests:
   - Test each method with valid/invalid inputs
   - Mock external dependencies
   - Edge cases: null, empty, boundary values
   
   Integration Tests:
   - Test component interactions
   - Database transactions
   - API contracts
   
   Performance Tests:
   - Load testing: sustained high traffic
   - Stress testing: peak capacity
   - Soak testing: memory leaks over time
   
   Chaos Engineering:
   - Random node failures
   - Network partitions
   - Clock skew
   
   Example: For LRU Cache:
   - Unit: put/get with full cache, empty cache
   - Integration: multi-threaded access
   - Performance: 1M operations/sec benchmark
   ```

5. **Database Design (30 mins)**
   
   **Common Question:** "What's your database schema?"
   
   **Your Approach:**
   ```
   1. Identify entities: Users, Orders, Products
   2. Define relationships: One-to-many, many-to-many
   3. Add indexes: Primary keys, foreign keys, query-based indexes
   4. Consider:
      - Normalization vs denormalization
      - Partitioning strategy for large tables
      - Read replicas for scaling reads
      - Caching layer (Redis) for hot data
   
   Example:
   CREATE TABLE users (
     user_id BIGINT PRIMARY KEY AUTO_INCREMENT,
     email VARCHAR(255) UNIQUE NOT NULL,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     INDEX idx_email (email)
   );
   ```

**🎯 Evening Session (90 mins): Company-Specific Prep**

6. **Research Your Target Company (45 mins)**
   
   - **If interviewing at Microsoft:**
     - Focus: Concurrency, Azure integration, C#/Java
     - Common problems: Threading, distributed systems
     - Study: Azure Service Bus, Cosmos DB patterns
   
   - **If interviewing at Amazon:**
     - Focus: Scalability, AWS services, leadership principles
     - Common problems: Distributed systems, high availability
     - Study: DynamoDB design, SQS/SNS patterns
   
   - **If interviewing at Google:**
     - Focus: Algorithms, scale, system efficiency
     - Common problems: Large-scale data processing
     - Study: Bigtable patterns, Spanner concepts
   
   - **If interviewing at Meta:**
     - Focus: Real-time systems, social graphs
     - Common problems: News feed, messaging, live updates
     - Study: Graph algorithms, pub-sub patterns

7. **Prepare Company-Specific Examples (45 mins)**
   - "How would you design [company product]?"
   - Microsoft: Teams presence system
   - Amazon: Product recommendation engine
   - Google: Search autocomplete
   - Meta: News feed ranking
   
   - Practice 10-minute explanation for your target company's product

**✅ Day 14 Success Criteria:**
- [ ] Can discuss security, monitoring, testing comprehensively
- [ ] Can analyze trade-offs at senior level
- [ ] Researched target company's technical focus
- [ ] Prepared company-specific examples

---

## Day 15 (Friday): Full Mock Interview Day

**Goal:** Simulate real interview conditions 3 times

**🎯 Mock Interview 1 (90 mins): Basic System**

**System:** LRU Cache with thread-safety
**Time:** 45 minutes of coding + 45 minutes of review

**Your Process:**
1. **Clarify (3 mins):** Capacity limits? Thread-safety needed? TTL support?
2. **Design (5 mins):** Draw classes, explain data structures
3. **Implement (25 mins):** Write code, narrate decisions
4. **Test (7 mins):** Walk through test cases, edge cases
5. **Follow-up (5 mins):** Handle "make it distributed" question

**Review Checklist:**
- [ ] Started with clarifying questions
- [ ] Explained before coding
- [ ] Handled edge cases
- [ ] Completed in time
- [ ] Handled follow-up confidently

**☕ Break (30 mins):** Review recording, note improvements

**🎯 Mock Interview 2 (90 mins): Distributed System**

**System:** Design a distributed message queue
**Time:** 45 minutes of design + 45 minutes of review

**Your Process:**
1. **Clarify (5 mins):** Throughput? Ordering guarantees? Persistence?
2. **High-Level (5 mins):** Components, data flow
3. **Deep Dive (20 mins):** Partition strategy, replication, consumer groups
4. **Failure Scenarios (10 mins):** Broker failure, network partition
5. **Optimization (5 mins):** Discuss batching, compression

**Review Checklist:**
- [ ] Covered functional requirements
- [ ] Discussed non-functional requirements (scalability, availability)
- [ ] Handled multiple failure scenarios
- [ ] Discussed trade-offs clearly
- [ ] Time management was good

**☕ Lunch Break (60 mins)**

**🎯 Mock Interview 3 (90 mins): Real-Time System**

**System:** Design a collaborative whiteboard
**Time:** 45 minutes of design + 45 minutes of review

**Your Process:**
1. **Clarify (5 mins):** Concurrent users? Conflict resolution? Offline mode?
2. **Architecture (8 mins):** WebSocket server, state sync, persistence
3. **Sync Algorithm (15 mins):** Operational Transform or CRDT
4. **Optimization (7 mins):** Caching, compression, delta updates
5. **Scale (10 mins):** Sharding users, load balancing

**Review Checklist:**
- [ ] Explained real-time sync strategy clearly
- [ ] Handled concurrent edit conflicts
- [ ] Discussed scaling to 1000 concurrent users
- [ ] Showed knowledge of WebSocket/long polling
- [ ] Communication was clear throughout

**🎯 Evening Session (60 mins): Reflection & Refinement**

**What Went Well:**
- [List 3 things you did well]

**What Needs Improvement:**
- [List 3 specific areas to work on]

**Action Items for Weekend:**
- [Specific practice tasks based on weaknesses]

**✅ Day 15 Success Criteria:**
- [ ] Completed 3 full mock interviews
- [ ] Reviewed and identified patterns in mistakes
- [ ] Have concrete action items for improvement
- [ ] Feeling more confident overall

---

## Weekend 3 (Sat-Sun): Final Polish & Mental Preparation

**🎯 Saturday (3 hours): Targeted Weakness Practice**

1. **Fix Your Top 3 Weaknesses (2 hours)**
   - Based on Friday's mocks, focus on:
   - **Weak area example 1:** Threading → Study thread-safe patterns, implement 3 concurrent systems
   - **Weak area example 2:** Communication → Record yourself explaining 5 systems, focus on clarity
   - **Weak area example 3:** Time management → Practice 3 problems with strict 30-min timer

2. **Cheat Sheet Finalization (1 hour)**
   - Create your personal 1-page cheat sheet:
   ```
   OPENING FRAMEWORK:
   - Clarify: Scale, Features, Constraints, Context
   - Design: High-level → Components → Interactions → Trade-offs
   - Implement: Talk through logic, handle edge cases
   - Test: Happy path, edge cases, stress scenarios
   
   COMMON PATTERNS:
   - Caching: LRU, distributed cache
   - Concurrency: Producer-consumer, thread pool
   - Partitioning: Consistent hashing
   - Replication: Quorum reads/writes
   
   REBUTTALS:
   - "Too simple" → Show layered thinking
   - "Won't scale" → Discuss partitioning/caching
   - "Why not X?" → Compare trade-offs
   - "Edge case?" → Validate and extend
   
   TIME MANAGEMENT:
   - 5 mins: Clarify
   - 5 mins: Design
   - 25 mins: Implement
   - 7 mins: Test
   - 3 mins: Follow-up buffer
   ```

**🎯 Sunday (2 hours): Mental Prep & Logistics**

1. **Final Review (60 mins)**
   - Skim through all 16 system implementations
   - Review your notes from 3 weeks
   - Don't learn anything new - just refresh

2. **Interview Logistics (30 mins)**
   - Test your setup: camera, mic, screen share
   - Prepare coding environment: IDE with templates
   - Have whiteboard/paper ready for drawing
   - Check time zone for interview
   - Set up quiet space

3. **Mental Preparation (30 mins)**
   - **Confidence Affirmations:**
     - "I've studied 16 systems thoroughly"
     - "I can handle any follow-up question"
     - "I communicate clearly under pressure"
     - "I know my trade-offs and can defend them"
   
   - **Stress Management:**
     - Deep breathing: 4 seconds in, 4 hold, 4 out
     - Visualize successful interview: walking through a problem confidently
     - Remember: Interviewers want you to succeed
   
   - **Day-Of Plan:**
     - Get good sleep (7-8 hours)
     - Light review in morning (30 mins max)
     - Eat a good meal before interview
     - Arrive (or log in) 10 minutes early
     - Take 5 deep breaths before starting

**✅ Weekend 3 Goal:**
- [ ] Fixed identified weaknesses
- [ ] Have 1-page cheat sheet ready
- [ ] Tested interview setup
- [ ] Feel mentally prepared and confident

---

# 🎯 INTERVIEW DAY STRATEGY

## The Day Of Your Interview

**2 Hours Before:**
- Light warm-up: Implement a simple LRU cache (15 mins)
- Review your cheat sheet (15 mins)
- Don't learn anything new - just prime your brain

**30 Minutes Before:**
- Test your setup: internet, video, audio, screen share
- Use bathroom, get water
- Do 5 minutes of deep breathing

**5 Minutes Before:**
- Close all distractions (Slack, email, phone)
- Open your coding environment with clean file
- Have whiteboard/paper ready for sketching
- Take 5 deep breaths

**During Interview:**

**First 30 Seconds (Critical):**
```
You: "Hi [Name], great to meet you! I'm excited to work on this problem together."
*Smile, show energy*
```

**Problem Presentation:**
```
You: "Let me make sure I understand... [restate problem]. 
Before I start designing, I have a few clarifying questions:
1. [Scale question]
2. [Feature question]
3. [Constraint question]
Is there anything else I should know about requirements?"
```

**During Coding:**
- Think out loud constantly
- If stuck for more than 30 seconds, say: "I'm considering between approach X and Y. X has advantage [...], Y has advantage [...]. I'll go with X because [reason]."
- Write TODO comments for complex parts, come back later

**When You Make a Mistake:**
```
You: "Actually, I see an issue here. [Explain problem]. Let me fix this..." 
*Calmly fix it*
```
**Don't:** Get flustered, apologize profusely, freeze

**When You Don't Know Something:**
```
You: "I'm not familiar with [specific thing], but I can reason through it. 
I would approach it by [reasonable guess], and I'd verify with documentation/colleagues. 
Does this direction seem reasonable?"
```
**Don't:** Say "I don't know" and stop

**Handling Interruptions:**
```
Interviewer: "What about [new requirement]?"
You: "Good point. Let me incorporate that. I can either [approach 1] or [approach 2]. Given this new requirement, I'd choose [approach 1] because [...]. Should I refactor now or finish current flow first?"
```

**Closing:**
```
You: "I've implemented the core functionality. Given time, I would add [error handling/optimization/testing]. Happy to dive into any area you'd like to explore further."
```

**After Interview:**
- Write down questions asked, what went well, what didn't
- Don't obsess - you did your preparation
- Follow up with thank you email within 24 hours

---

# 🎯 REBUTTAL SCRIPTS - MEMORIZE THESE

## Response Templates for Common Pushback

### "This Design is Too Simple"

**Template:**
```
"You're absolutely right that production systems need [missing component]. 
I started with core functionality to validate the approach and ensure correctness. 
Let me extend this to include [component]:
[Proceeds to add complexity]

The trade-off here is [simple vs complex trade-off]. 
Given [requirement/constraint], this level of complexity is appropriate. 
Should we go deeper into [specific area]?"
```

**Example:**
```
Interviewer: "Your cache is too simple. What about cache invalidation?"
You: "You're right, production caches need sophisticated invalidation. 
I started with core get/put to validate the LRU algorithm. 
Let me add invalidation strategies:

1. Time-based: TTL per entry (add expiry_time field)
2. Event-based: Publish-subscribe for invalidation messages
3. Manual: Explicit invalidate() method

For this use case [ask if not clear], I'd use TTL with lazy cleanup 
because it's simpler and handles most cases. For stricter consistency, 
I'd add event-based invalidation. Should I implement one of these?"
```

### "This Won't Scale"

**Template:**
```
"Agreed, single-node design has scaling limits. For [mentioned scale], I would:
1. [Horizontal scaling technique] - specific benefit
2. [Caching layer] - specific benefit
3. [Async processing] - specific benefit

The bottleneck here is likely [identify bottleneck]. 
By [specific technique], we can handle [target scale].
Which area would you like me to detail?"
```

**Example:**
```
Interviewer: "This won't handle 10,000 requests per second."
You: "Agreed, single instance has limits around [reasonable number] req/sec. 
For 10K req/sec, I would:

1. Horizontal scaling: Partition data using consistent hashing across N nodes
2. Read replicas: Route reads to replicas (assuming read-heavy workload)
3. Caching: Add Redis layer for hot keys (80/20 rule - 80% hits on 20% keys)
4. Load balancing: HAProxy/nginx in front for distribution

The main bottleneck is disk I/O for writes. With partitioning, each node handles 
10K/N writes, making it manageable. For extreme scale, we'd add async writes with 
write-ahead log.

Should I sketch the distributed architecture?"
```

### "Why Not Use [Alternative Approach]?"

**Template:**
```
"[Alternative] is a valid choice. Let me compare trade-offs:

My Approach ([Current]):
✓ Advantage 1: [specific benefit]
✓ Advantage 2: [specific benefit]
✗ Disadvantage: [specific drawback]

Alternative ([Suggested]):
✓ Advantage 1: [specific benefit]
✗ Disadvantage 1: [specific drawback]
✗ Disadvantage 2: [specific drawback]

Given requirement [specific requirement], I chose [current] because [reason]. 
However, if [condition changes], [alternative] would be better. 
Would you like me to pivot to [alternative]?"
```

**Example:**
```
Interviewer: "Why HashMap? Why not just use a database?"
You: "Database is valid for persistence. Let me compare:

HashMap (my choice):
✓ O(1) lookups - critical for low latency requirement
✓ In-memory - no disk I/O overhead
✓ Simple - no schema management
✗ No persistence - data lost on restart
✗ Limited by RAM - can't store unlimited data

Database:
✓ Persistence - survives restarts
✓ Query flexibility - SQL for complex queries
✗ Slower - disk I/O adds ~10ms latency
✗ More complex - connection pooling, transactions

Given the requirement for sub-millisecond latency and the ephemeral nature 
of cache data, HashMap is appropriate. If we needed persistence, I'd add 
write-ahead log or use Redis (in-memory DB with persistence).

Does this align with requirements, or should we prioritize persistence?"
```

### "What About [Edge Case]?"

**Template:**
```
"Good catch. [Acknowledge the edge case specifics].

Current handling: [Explain if already handled, or admit gap]

To handle this properly, I would [specific solution]:
[Show code or design change]

The trade-off is [any performance/complexity cost].

Are there other edge cases I should consider?"
```

**Example:**
```
Interviewer: "What if two users try to book the same parking spot simultaneously?"
You: "Excellent question - that's a classic race condition.

Current code has a race: 
1. User A checks spot available (true)
2. User B checks spot available (true)  
3. Both try to book same spot

To fix, I'd add pessimistic locking:

def book_spot(self, spot_id):
    with self.lock:  # Acquire lock
        spot = self.spots[spot_id]
        if not spot.is_available:
            return None  # Already booked
        spot.is_available = False
        ticket = self.create_ticket(spot)
        return ticket
    # Lock released

Alternative: Optimistic locking with version numbers if we want 
less contention on popular spots.

Trade-off: Locking reduces throughput under high contention. 
For parking lot (low booking frequency), pessimistic lock is fine.

Should I also add timeout for reservations?"
```

### "We're Running Out of Time"

**Template:**
```
"Understood. Let me prioritize. Currently, I have:
✅ [What's complete]
⏳ [What's in progress]

I can either:
Option 1: [Finish current task] - gives us [specific benefit]
Option 2: [Add critical feature] - covers [important requirement]
Option 3: [Discuss optimization] - shows [scaling knowledge]

What would be most valuable to you?"
```

**Example:**
```
Interviewer: "We have 5 minutes left."
You: "Understood. Currently, I have:
✅ Core LRU cache with get/put operations
✅ Capacity management with eviction
⏳ Working on thread-safety

I can either:
1. Complete thread-safety implementation - makes it production-ready
2. Add TTL-based expiration - shows distributed cache knowledge
3. Discuss how to scale this to distributed cache - demonstrates system design thinking

What would be most valuable in remaining time?"

[Interviewer chooses, or:]
"If you're good with current state, I can walk through how I'd test this 
and what metrics I'd track in production."
```

---

# 📊 FINAL CHECKLIST

## Day Before Interview

**Technical Review:**
- [ ] Skim through 5 most common systems (Cache, Rate Limiter, Parking, URL Shortener, Elevator)
- [ ] Review your cheat sheet
- [ ] Don't practice new problems - just refresh memory

**Logistics:**
- [ ] Test camera, microphone, internet connection
- [ ] Prepare coding environment (IDE with favorite settings)
- [ ] Have whiteboard/paper ready for diagrams
- [ ] Confirm interview time (check time zones!)
- [ ] Know interviewer names if provided

**Physical/Mental:**
- [ ] Get 7-8 hours of sleep
- [ ] Prepare comfortable clothes
- [ ] Set up quiet space (tell roommates/family)
- [ ] Prepare water and snacks nearby

## Day of Interview

**Morning Of:**
- [ ] Light warm-up problem (30 mins max)
- [ ] Review cheat sheet (10 mins)
- [ ] Eat a good meal
- [ ] Avoid caffeine overload (normal amount OK)

**30 Minutes Before:**
- [ ] Final setup test: video, audio, screen share
- [ ] Close all distractions (Slack, email, notifications)
- [ ] Open clean coding environment
- [ ] Use bathroom, get water
- [ ] 5 minutes of deep breathing

**During Interview:**
- [ ] Smile and show energy in greeting
- [ ] Take notes on requirements
- [ ] Ask clarifying questions before coding
- [ ] Think out loud constantly
- [ ] Handle pushback gracefully
- [ ] Manage time - check clock every 10 mins
- [ ] Ask questions at end if offered

**After Interview:**
- [ ] Write down questions asked
- [ ] Note what went well and what didn't
- [ ] Don't obsess - you prepared well
- [ ] Send thank you email within 24 hours

---

# 🏆 KEY TAKEAWAYS - THE REAL SECRETS

## What Actually Matters in LLD Interviews

**1. Communication > Perfect Code**
- Interviewers care more about HOW you think than perfect syntax
- Talk through your approach before coding
- Explain trade-offs - "I chose X over Y because..."
- Ask questions when unsure, don't guess silently

**2. Structure > Speed**
- Taking 5 minutes to design properly saves 20 minutes of refactoring
- Always: Clarify → Design → Implement → Test
- Use consistent patterns across problems
- Show you can think before you code

**3. Trade-offs > Single Solution**
- There's no "right answer" - only trade-offs
- "This approach prioritizes X over Y because..."
- Show you can evaluate multiple options
- Adapt based on requirements

**4. Handling Pushback > Initial Design**
- Interviews test how you handle criticism
- Stay calm, acknowledge points, adapt
- "That's a valid concern, let me address it..."
- Show you can iterate based on feedback

**5. Breadth > Depth**
- Better to cover all requirements at 80% than one at 100%
- Time management is critical
- "I can add [feature] if time permits"
- Know when to move on

## Common Mistakes to Avoid

❌ **Jumping to code without clarifying**
✅ Spend 5 minutes on requirements

❌ **Silent coding**
✅ Narrate your thought process

❌ **Getting defensive about design**
✅ "That's a good point, let me adjust..."

❌ **Ignoring edge cases**
✅ Proactively mention: "Edge cases to handle: null input, full capacity..."

❌ **Over-engineering initially**
✅ Start simple, add complexity when asked

❌ **Saying "I don't know" and stopping**
✅ "I'm not sure, but here's how I'd approach it..."

❌ **Not asking about requirements**
✅ "What's the expected scale? Thread-safety needed?"

## Your Competitive Advantages

After 3 weeks, you have:
1. **16 systems** in muscle memory
2. **Rebuttal scripts** for pushback
3. **Opening framework** for any problem
4. **Pattern recognition** across domains
5. **Communication skills** from mocks
6. **Confidence** from preparation

**You are ready. Trust your preparation.**

---

# 📞 EMERGENCY SCENARIOS

## If Things Go Wrong

### Technical Difficulties

**Screen share not working:**
"I'm having issues with screen share. Can I describe my code verbally while I troubleshoot, or should we reschedule?"

**Internet drops:**
Have phone number ready to call immediately. "Sorry, my internet dropped. I'm back now."

**IDE crashes:**
Have a backup plan: online IDE like repl.it or even Google Docs for pseudocode

### Mental Blocks

**Completely stuck:**
"I'm considering a few approaches. Let me think out loud - I could use [approach 1] which gives [benefit], or [approach 2] which handles [case] better. Which direction seems more aligned with requirements?"

**Forgot algorithm:**
"I know the general principle of [algorithm] - [describe at high level]. For production, I'd verify exact implementation details, but here's the core logic..."

**Misunderstood problem:**
"Actually, let me make sure I understand correctly. You're asking for [rephrase]? I initially understood it as [wrong understanding]. Let me adjust my approach..."

### Interviewer Challenges

**Hostile interviewer:**
Stay professional, don't get emotional. "I understand your concern about [point]. Let me address it with [solution]."

**Constantly interrupted:**
"I appreciate the feedback. Let me complete this thought, then I'll incorporate your suggestion."

**Unclear requirements:**
"I want to make sure I'm building the right thing. Could you clarify [specific ambiguity]?"

---

# 💪 CONFIDENCE BUILDERS

## Remind Yourself

**You've Done the Work:**
- Studied 16 complete systems
- Completed 6+ mock interviews
- Practiced rebuttals for every scenario
- Understand patterns across domains

**You Have the Skills:**
- Can implement LRU cache in 25 minutes
- Know distributed systems patterns
- Can handle concurrency correctly
- Communicate clearly under pressure

**You're Well-Prepared:**
- More prepared than 90% of candidates
- Have structured frameworks, not just knowledge
- Practiced under realistic conditions
- Ready for any follow-up question

**Remember:**
- Interviewers want you to succeed (they're hiring!)
- Mistakes are OK if you handle them well
- It's a conversation, not an interrogation
- You're evaluating them too

---

# 🎯 FINAL WORDS

You've completed an intensive 3-week bootcamp. You've:

✅ Mastered 16 production-ready systems
✅ Learned advanced patterns and distributed concepts
✅ Practiced under realistic interview conditions
✅ Developed communication and rebuttal skills
✅ Built confidence through preparation

**You are ready for your LLD interview.**

Walk in (or log on) with confidence. You know your stuff. You can handle any question. You've prepared more thoroughly than most candidates ever will.

**Trust your preparation. Communicate clearly. Handle pushback gracefully. You've got this.**

---

## Good luck! 🚀

*Now go ace that interview and come back to share your success story.*

---

**Remember:** This guide is comprehensive, but the real learning happens when YOU implement systems and practice explaining them. Don't just read - code, practice, iterate.

**Next Steps:**
1. Print your 1-page cheat sheet
2. Do one final mock interview with a friend/recording
3. Get good sleep the night before
4. Execute your interview with confidence

**You've prepared. Now perform. You've got this! 💪**

```python
# Encapsulation - Hide internal details
class BankAccount:
    def __init__(self):
        self.__balance = 0  # Private
    
    def deposit(self, amount):
        self.__balance += amount

# Inheritance - Code reuse
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

# Polymorphism - Same interface, different implementations
class Shape:
    def area(self):
        pass

class Circle(Shape):
    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def area(self):
        return self.width * self.height

# Abstraction - Hide complexity
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass

class StripeProcessor(PaymentProcessor):
    def process_payment(self, amount):
        # Stripe-specific implementation
        pass
```

</details>

<details>
<summary><b>✍️ Hands-On Exercise (1 hour)</b></summary>

**Exercise 1: Implement a Simple HashMap**

```python
class SimpleHashMap:
    def __init__(self, size=10):
        self.size = size
        self.buckets = [[] for _ in range(size)]
    
    def _hash(self, key):
        # TODO: Implement hash function
        pass
    
    def put(self, key, value):
        # TODO: Handle collisions with chaining
        pass
    
    def get(self, key):
        # TODO: Return value or None
        pass
    
    def remove(self, key):
        # TODO: Remove key-value pair
        pass

# Test your implementation
hashmap = SimpleHashMap()
hashmap.put("name", "Alice")
assert hashmap.get("name") == "Alice"
hashmap.remove("name")
assert hashmap.get("name") is None
```

**Exercise 2: Implement a Doubly Linked List**

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def add_first(self, data):
        # TODO: Add to front - O(1)
        pass
    
    def add_last(self, data):
        # TODO: Add to end - O(1)
        pass
    
    def remove_first(self):
        # TODO: Remove from front - O(1)
        pass
    
    def remove_last(self):
        # TODO: Remove from end - O(1)
        pass

# Test cases
dll = DoublyLinkedList()
dll.add_last(1)
dll.add_last(2)
dll.add_first(0)
# List should be: 0 -> 1 -> 2
```

</details>

<details>
<summary><b>🎯 Self-Assessment Quiz</b></summary>

**Question 1**: What's the time complexity of searching in a HashMap?  
**Answer**: O(1) average case, O(n) worst case with many collisions

**Question 2**: Why use a Doubly Linked List over a Singly Linked List?  
**Answer**: O(1) deletion from tail, bidirectional traversal

**Question 3**: When should you use a Heap vs a Sorted Array?  
**Answer**: Heap for frequent insertions (O(log n)), Sorted Array for frequent searches (O(log n))

**Question 4**: What's the difference between Abstract Class and Interface?  
**Answer**: Abstract class can have implementation, Interface is pure contract (Python uses ABC for both)

**Question 5**: How do you prevent tight coupling in OOP?  
**Answer**: Dependency injection, interfaces, composition over inheritance

</details>

<details>
<summary><b>📌 Key Takeaways</b></summary>

✅ **Master these data structures**: HashMap, LinkedList, Heap, Tree  
✅ **OOP is about design, not just syntax**  
✅ **Time/Space complexity matters in interviews**  
✅ **Always consider trade-offs** (speed vs memory)  

**Tomorrow**: Apply these concepts to build your first system (Cache)!

</details>

---

### Day 2: Cache System - Your First LLD Implementation

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Implement LRU Cache from scratch in 30 minutes
- [ ] Explain why HashMap + Doubly Linked List combination works
- [ ] Handle edge cases (empty cache, capacity 1, etc.)
- [ ] Discuss thread safety considerations

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**Step 1: Read the Full Implementation**

Open `01_cache_system.py` and `01_cache_system_readme.md`. Don't just skim—actually trace through the code:

```python
# Key insight for LRU Cache:
# 1. HashMap: key -> Node (O(1) lookup)
# 2. Doubly Linked List: maintain access order
# 3. Most recent at tail, least recent at head

# When get(key):
#   - Move node to tail (most recent)
#   - Return value
#
# When put(key, value):
#   - Add to tail
#   - If over capacity, remove head
```

**Step 2: Understand the Pattern**

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        self.head = Node(0, 0)  # Dummy head
        self.tail = Node(0, 0)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
    
    def _remove(self, node):
        # Remove from linked list
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add(self, node):
        # Add to tail (most recent)
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node
```

**Step 3: Watch It Work**

Run `python3 01_cache_system.py` and observe the output. Add print statements to trace execution:

```python
def get(self, key):
    print(f"Getting key: {key}")
    print(f"Cache state: {list(self.cache.keys())}")
    # ... rest of code
```

</details>

<details>
<summary><b>✍️ Coding Practice (2 hours)</b></summary>

**Exercise 1: Implement from Scratch (45 min)**

Close all references and implement LRU Cache from memory:

```python
# Your turn - implement without looking!

class Node:
    # TODO: Define node for doubly linked list
    pass

class LRUCache:
    def __init__(self, capacity):
        # TODO: Initialize data structures
        pass
    
    def get(self, key):
        # TODO: Return value and update recency
        pass
    
    def put(self, key, value):
        # TODO: Add/update and evict if needed
        pass

# Test cases
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)  # Evicts key 2
assert cache.get(2) == -1
```

**Exercise 2: Implement LFU Cache (45 min)**

Now try Least Frequently Used:

```python
class LFUCache:
    """
    Evict the least frequently used item.
    Tie-breaker: least recently used among them.
    """
    
    def __init__(self, capacity):
        # TODO: Need to track frequency
        # Hint: Use freq -> LinkedList mapping
        pass
    
    def get(self, key):
        # TODO: Increment frequency
        pass
    
    def put(self, key, value):
        # TODO: Handle frequency updates and eviction
        pass
```

**Exercise 3: Add TTL (Time-To-Live) (30 min)**

Enhance your LRU Cache with expiration:

```python
import time

class TTLCache(LRUCache):
    def __init__(self, capacity, default_ttl=60):
        super().__init__(capacity)
        self.default_ttl = default_ttl
        self.expiry = {}  # key -> expiry_timestamp
    
    def put(self, key, value, ttl=None):
        # TODO: Store expiry time
        pass
    
    def get(self, key):
        # TODO: Check if expired before returning
        pass
    
    def _cleanup_expired(self):
        # TODO: Remove expired entries
        pass
```

</details>

<details>
<summary><b>🎯 Mock Interview Question</b></summary>

**Interviewer**: "Design a cache system for a web application."

**Expected Response Structure:**

```
1. Clarification (2 min):
   - Q: "What's the expected number of entries?"
   - Q: "Read-heavy or write-heavy workload?"
   - Q: "Need distributed or single-server?"
   - Q: "Any eviction policy preference?"

2. High-Level Design (3 min):
   - Components: Cache interface, Storage, Eviction policy
   - API: get(key), put(key, value), delete(key)
   - Choose LRU for balanced performance

3. Deep Dive (20 min):
   - Implement LRUCache class
   - Explain HashMap + DLL approach
   - Handle edge cases
   - Discuss thread safety

4. Extensions (5 min):
   - TTL for automatic expiration
   - Write-through vs write-back
   - Distributed caching with Redis
   - Cache warming strategies
```

**Model Answer for Thread Safety:**

```python
import threading

class ThreadSafeLRUCache:
    def __init__(self, capacity):
        self.cache = LRUCache(capacity)
        self.lock = threading.RLock()  # Reentrant lock
    
    def get(self, key):
        with self.lock:
            return self.cache.get(key)
    
    def put(self, key, value):
        with self.lock:
            self.cache.put(key, value)

# Why RLock? A thread can acquire it multiple times
# Important for nested method calls
```

</details>

<details>
<summary><b>✅ Day 2 Checkpoint</b></summary>

Before moving on, verify you can:
- [ ] Implement LRU Cache in < 30 minutes
- [ ] Explain time complexity (O(1) for both get/put)
- [ ] Handle edge cases without bugs
- [ ] Discuss at least 2 eviction policies
- [ ] Explain thread safety solutions

**If stuck**: Re-read the implementation, trace with debugger, watch YouTube tutorials

**Tomorrow**: Parking Lot System (OOP design patterns)

</details>

---

### Day 3: Parking Lot System - OOP Mastery

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Design class hierarchies for real-world entities
- [ ] Apply Singleton pattern correctly
- [ ] Handle pricing calculations with strategy pattern
- [ ] Model relationships (composition, aggregation)

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**Core Concepts:**

```python
# 1. Entity Modeling
# Real-world objects become classes

class ParkingSpot:
    """Represents a physical parking spot"""
    def __init__(self, spot_id, spot_type, floor):
        self.spot_id = spot_id
        self.spot_type = spot_type  # COMPACT, LARGE, HANDICAPPED
        self.floor = floor
        self.is_occupied = False
        self.vehicle = None

class Vehicle:
    """Represents a vehicle"""
    def __init__(self, license_plate, vehicle_type):
        self.license_plate = license_plate
        self.vehicle_type = vehicle_type  # CAR, TRUCK, MOTORCYCLE

# 2. Relationships
class ParkingLot:
    """Has-A relationship with ParkingSpots"""
    def __init__(self):
        self.floors = []
        self.spots = {}  # spot_id -> ParkingSpot
        self.tickets = {}  # ticket_id -> ParkingTicket

class ParkingTicket:
    """Association with Vehicle and ParkingSpot"""
    def __init__(self, ticket_id, vehicle, spot):
        self.ticket_id = ticket_id
        self.vehicle = vehicle
        self.spot = spot
        self.entry_time = datetime.now()
        self.exit_time = None

# 3. Singleton Pattern for Controller
class ParkingLotController:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.parking_lot = ParkingLot()
            self.initialized = True
```

**Design Patterns Applied:**

1. **Singleton**: Ensure only one ParkingLotController exists
2. **Strategy**: Different pricing strategies (hourly, daily, flat)
3. **Factory**: Create different types of spots/vehicles
4. **Observer**: Notify when spots become available

</details>

<details>
<summary><b>✍️ Hands-On Exercise (2 hours)</b></summary>

**Exercise 1: Implement Parking Spot Finder**

```python
from enum import Enum
from typing import Optional

class SpotType(Enum):
    COMPACT = 1
    LARGE = 2
    HANDICAPPED = 3
    MOTORCYCLE = 4

class VehicleType(Enum):
    CAR = 1
    TRUCK = 2
    MOTORCYCLE = 3

class SpotFinder:
    """Find optimal parking spot for vehicle"""
    
    @staticmethod
    def find_spot(parking_lot, vehicle_type) -> Optional[ParkingSpot]:
        """
        TODO: Implement spot finding logic
        
        Rules:
        - MOTORCYCLE can park in MOTORCYCLE or COMPACT
        - CAR can park in COMPACT or LARGE
        - TRUCK needs LARGE spot
        - HANDICAPPED spots for handicapped vehicles only
        
        Strategy: Find nearest available spot
        """
        pass

# Test your implementation
lot = ParkingLot()
# Add some spots...
car = Vehicle("ABC123", VehicleType.CAR)
spot = SpotFinder.find_spot(lot, car.vehicle_type)
assert spot is not None
```

**Exercise 2: Implement Pricing Calculator**

```python
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, entry_time: datetime, exit_time: datetime) -> float:
        pass

class HourlyPricing(PricingStrategy):
    def __init__(self, rate_per_hour: float):
        self.rate = rate_per_hour
    
    def calculate_price(self, entry_time, exit_time):
        # TODO: Calculate based on hours
        # Round up partial hours
        pass

class FlatRatePricing(PricingStrategy):
    def __init__(self, flat_rate: float, max_hours: int):
        self.flat_rate = flat_rate
        self.max_hours = max_hours
    
    def calculate_price(self, entry_time, exit_time):
        # TODO: Flat rate up to max_hours, then hourly
        pass

class DynamicPricing(PricingStrategy):
    def __init__(self):
        self.peak_hours = [(7, 10), (17, 20)]  # 7-10am, 5-8pm
        self.peak_rate = 5.0
        self.regular_rate = 3.0
    
    def calculate_price(self, entry_time, exit_time):
        # TODO: Calculate with peak/regular rates
        pass

# Test cases
hourly = HourlyPricing(3.0)
entry = datetime(2024, 1, 1, 10, 0)
exit = datetime(2024, 1, 1, 12, 30)
price = hourly.calculate_price(entry, exit)
assert price == 9.0  # 3 hours * $3
```

**Exercise 3: Implement Reservation System**

```python
class Reservation:
    def __init__(self, vehicle, start_time, duration_hours):
        self.vehicle = vehicle
        self.start_time = start_time
        self.end_time = start_time + timedelta(hours=duration_hours)
        self.spot = None
        self.status = "PENDING"  # PENDING, CONFIRMED, CANCELLED

class ReservationManager:
    def __init__(self, parking_lot):
        self.parking_lot = parking_lot
        self.reservations = {}  # reservation_id -> Reservation
    
    def make_reservation(self, vehicle, start_time, duration):
        """
        TODO: Reserve a spot in advance
        - Check if spot will be available
        - Block the spot for this time window
        - Handle conflicts
        """
        pass
    
    def cancel_reservation(self, reservation_id):
        # TODO: Free up the spot
        pass
    
    def check_in(self, reservation_id):
        # TODO: Convert reservation to active ticket
        pass
```

</details>

<details>
<summary><b>🎯 Design Discussion Questions</b></summary>

**Q1**: Why use Enum for SpotType instead of strings?  
**A**: Type safety, autocomplete, prevent typos, better refactoring

**Q2**: When should you use Singleton pattern?  
**A**: When exactly one instance is needed (controller, config manager), but be cautious—often composition is better

**Q3**: How would you handle concurrent booking of same spot?  
**A**: Database transactions, optimistic locking, or pessimistic locking with timeout

**Q4**: How to extend for multi-floor parking?  
**A**: Add Floor class, organize spots by floor, find nearest floor with available spots

**Q5**: How to handle monthly pass holders?  
**A**: Add PassHolder class, check before pricing, reserve percentage of spots

</details>

<details>
<summary><b>✅ Day 3 Checkpoint</b></summary>

- [ ] Designed 5+ classes with clear responsibilities
- [ ] Applied Singleton pattern correctly
- [ ] Implemented Strategy pattern for pricing
- [ ] Handled edge cases (full lot, invalid vehicle type)
- [ ] Can explain class relationships

**Tomorrow**: Library Management System (business logic)

</details>

---

### Day 4: Library Management System - Business Logic

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Design a complete library management system
- [ ] Implement book checkout/return with due dates
- [ ] Calculate fines for late returns
- [ ] Handle member types with different privileges
- [ ] Implement search functionality (by title, author, ISBN)

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**System Requirements:**

```python
"""
Library Management System Components:

1. Entities:
   - Book (title, author, ISBN, copies)
   - Member (name, member_id, type)
   - Librarian (extends Member)
   - Transaction (checkout/return record)

2. Business Rules:
   - Regular members: 5 books max, 14 days
   - Premium members: 10 books max, 30 days
   - Students: 3 books max, 7 days
   - Fine: $1/day for late returns
   - Reserved books: hold for 3 days

3. Operations:
   - Search books (title/author/ISBN)
   - Checkout book
   - Return book (calculate fine)
   - Reserve book
   - Renew book (if not reserved by others)
"""

from datetime import datetime, timedelta
from enum import Enum

class MemberType(Enum):
    REGULAR = "regular"
    PREMIUM = "premium"
    STUDENT = "student"

class BookStatus(Enum):
    AVAILABLE = "available"
    CHECKED_OUT = "checked_out"
    RESERVED = "reserved"

class Book:
    def __init__(self, isbn, title, author, total_copies):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.total_copies = total_copies
        self.available_copies = total_copies
        self.status = BookStatus.AVAILABLE
    
    def checkout(self):
        if self.available_copies > 0:
            self.available_copies -= 1
            return True
        return False
    
    def return_book(self):
        self.available_copies += 1

class Member:
    def __init__(self, member_id, name, member_type):
        self.member_id = member_id
        self.name = name
        self.member_type = member_type
        self.checked_out_books = []
        
        # Set limits based on type
        self.limits = {
            MemberType.REGULAR: {"max_books": 5, "days": 14},
            MemberType.PREMIUM: {"max_books": 10, "days": 30},
            MemberType.STUDENT: {"max_books": 3, "days": 7}
        }
    
    def can_checkout(self):
        limit = self.limits[self.member_type]["max_books"]
        return len(self.checked_out_books) < limit
    
    def get_borrow_period(self):
        return self.limits[self.member_type]["days"]

class Transaction:
    def __init__(self, transaction_id, member, book):
        self.transaction_id = transaction_id
        self.member = member
        self.book = book
        self.checkout_date = datetime.now()
        self.due_date = self.checkout_date + timedelta(
            days=member.get_borrow_period()
        )
        self.return_date = None
        self.fine = 0.0
    
    def calculate_fine(self, return_date):
        """$1 per day for late returns"""
        if return_date > self.due_date:
            days_late = (return_date - self.due_date).days
            self.fine = days_late * 1.0
        return self.fine
```

</details>

<details>
<summary><b>✍️ Hands-On Exercise (2 hours)</b></summary>

**Exercise 1: Implement Library Catalog (45 min)**

```python
class LibraryCatalog:
    def __init__(self):
        self.books = {}  # ISBN -> Book
        self.by_title = {}  # title -> [ISBNs]
        self.by_author = {}  # author -> [ISBNs]
    
    def add_book(self, book):
        """
        TODO: Add book to catalog
        - Store in books dict
        - Index by title
        - Index by author
        """
        pass
    
    def search_by_title(self, title):
        """TODO: Return list of matching books"""
        pass
    
    def search_by_author(self, author):
        """TODO: Return list of books by author"""
        pass
    
    def search_by_isbn(self, isbn):
        """TODO: Return specific book"""
        pass
    
    def get_available_books(self):
        """TODO: Return books with available copies > 0"""
        pass

# Test
catalog = LibraryCatalog()
book1 = Book("978-0-13-468599-1", "Clean Code", "Robert Martin", 5)
book2 = Book("978-0-201-63361-0", "Design Patterns", "Gang of Four", 3)
catalog.add_book(book1)
catalog.add_book(book2)

results = catalog.search_by_title("Clean Code")
assert len(results) == 1
assert results[0].author == "Robert Martin"
```

**Exercise 2: Implement Checkout/Return System (45 min)**

```python
class LibraryManager:
    def __init__(self, catalog):
        self.catalog = catalog
        self.transactions = {}  # transaction_id -> Transaction
        self.member_checkouts = {}  # member_id -> [transaction_ids]
        self.next_transaction_id = 1
    
    def checkout_book(self, member, isbn):
        """
        TODO: Checkout book
        - Check if member can checkout (not at limit)
        - Check if book available
        - Create transaction
        - Update book availability
        - Record in member's checkouts
        """
        pass
    
    def return_book(self, transaction_id):
        """
        TODO: Return book
        - Find transaction
        - Calculate fine if late
        - Update book availability
        - Remove from member's checkouts
        - Return fine amount
        """
        pass
    
    def renew_book(self, transaction_id):
        """
        TODO: Extend due date
        - Only if not reserved by others
        - Extend by member's borrow period
        """
        pass
    
    def get_overdue_books(self):
        """TODO: Return list of overdue transactions"""
        pass

# Test
manager = LibraryManager(catalog)
member = Member("M001", "Alice", MemberType.REGULAR)

# Checkout
transaction_id = manager.checkout_book(member, "978-0-13-468599-1")
assert transaction_id is not None

# Return after 20 days (6 days late, $6 fine)
# Simulate by setting return date manually
transaction = manager.transactions[transaction_id]
transaction.return_date = transaction.due_date + timedelta(days=6)
fine = manager.return_book(transaction_id)
assert fine == 6.0
```

**Exercise 3: Implement Reservation System (30 min)**

```python
class ReservationSystem:
    def __init__(self, library_manager):
        self.library_manager = library_manager
        self.reservations = {}  # isbn -> queue of member_ids
    
    def reserve_book(self, member_id, isbn):
        """
        TODO: Reserve book
        - Add to queue
        - Set expiry (3 days from availability)
        """
        pass
    
    def notify_available(self, isbn):
        """
        TODO: Notify next person in queue
        - Send notification (print for now)
        - Set 3-day hold period
        """
        pass
    
    def cancel_reservation(self, member_id, isbn):
        """TODO: Remove from queue"""
        pass
```

</details>

<details>
<summary><b>🎯 Self-Assessment Quiz</b></summary>

**Q1**: How do you prevent a member from checking out more books than their limit?  
**A**: Check `len(member.checked_out_books) < limit` before checkout

**Q2**: How to handle partial title search (e.g., "Clean" matches "Clean Code")?  
**A**: Use lowercase and `in` operator, or implement Trie for better performance

**Q3**: What data structure for fast ISBN lookup?  
**A**: HashMap (O(1)), but could use TreeMap if need sorting

**Q4**: How to efficiently find all overdue books?  
**A**: Maintain sorted list by due date, or scan transactions (acceptable for small scale)

**Q5**: How to handle book reservations fairly?  
**A**: Queue (FIFO), with expiry mechanism (3 days to checkout)

</details>

<details>
<summary><b>✅ Day 4 Checkpoint</b></summary>

- [ ] Implemented book catalog with search
- [ ] Checkout/return with fine calculation working
- [ ] Different member types with limits
- [ ] Can explain business logic clearly
- [ ] Handled edge cases (no copies, over limit)

**Tomorrow**: Snake & Ladder Game (Strategy pattern)

</details>

---

### Day 5: Snake & Ladder Game - Strategy Pattern

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Implement Strategy pattern correctly
- [ ] Design game state management
- [ ] Handle turn-based gameplay
- [ ] Track game statistics
- [ ] Test with controlled inputs (ControlledDice)

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**Strategy Pattern Deep Dive:**

```python
"""
Strategy Pattern: Define family of algorithms,
encapsulate each one, make them interchangeable.

Why useful for Snake & Ladder:
1. Testing: Use ControlledDice with predetermined rolls
2. Variations: WeightedDice for modified gameplay
3. AI: SmartDice that can learn patterns

Components:
- Strategy Interface: DiceStrategy
- Concrete Strategies: StandardDice, WeightedDice, ControlledDice
- Context: Game uses strategy without knowing which one
"""

from abc import ABC, abstractmethod
import random

# Strategy Interface
class DiceStrategy(ABC):
    @abstractmethod
    def roll(self):
        """Return dice roll value"""
        pass

# Concrete Strategy 1: Standard 6-sided dice
class StandardDice(DiceStrategy):
    def roll(self):
        return random.randint(1, 6)

# Concrete Strategy 2: Weighted dice (higher numbers more likely)
class WeightedDice(DiceStrategy):
    def roll(self):
        # 1-2: 10% each, 3-4: 15% each, 5-6: 25% each
        return random.choices(
            [1, 2, 3, 4, 5, 6],
            weights=[10, 10, 15, 15, 25, 25]
        )[0]

# Concrete Strategy 3: Controlled dice (for testing)
class ControlledDice(DiceStrategy):
    def __init__(self, rolls):
        self.rolls = rolls
        self.index = 0
    
    def roll(self):
        if self.index >= len(self.rolls):
            self.index = 0  # Loop
        value = self.rolls[self.index]
        self.index += 1
        return value

# Game Board with Snake/Ladder mappings
class Board:
    def __init__(self, size=100):
        self.size = size
        self.snakes = {}  # head -> tail
        self.ladders = {}  # bottom -> top
    
    def add_snake(self, head, tail):
        """Add snake from head to tail"""
        if head > tail and head <= self.size:
            self.snakes[head] = tail
    
    def add_ladder(self, bottom, top):
        """Add ladder from bottom to top"""
        if bottom < top and top <= self.size:
            self.ladders[bottom] = top
    
    def get_final_position(self, position):
        """Check for snake/ladder at position"""
        if position in self.snakes:
            return self.snakes[position]
        if position in self.ladders:
            return self.ladders[position]
        return position
    
    def setup_classic(self):
        """Setup classic 100-square board"""
        # Snakes
        snakes = [(99, 54), (70, 55), (52, 42), (25, 2), (95, 72)]
        for head, tail in snakes:
            self.add_snake(head, tail)
        
        # Ladders
        ladders = [(6, 40), (23, 56), (45, 83), (61, 99), (21, 82)]
        for bottom, top in ladders:
            self.add_ladder(bottom, top)

# Player with statistics
class Player:
    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name
        self.position = 0
        self.moves_count = 0
        self.snakes_hit = 0
        self.ladders_climbed = 0
    
    def move(self, steps, board):
        """Move player and handle snake/ladder"""
        new_position = self.position + steps
        
        # Don't move if overshooting
        if new_position > board.size:
            return self.position
        
        # Check for snake/ladder
        final_position = board.get_final_position(new_position)
        
        # Update stats
        if new_position in board.snakes:
            self.snakes_hit += 1
        elif new_position in board.ladders:
            self.ladders_climbed += 1
        
        self.position = final_position
        self.moves_count += 1
        return final_position
```

</details>

<details>
<summary><b>✍️ Hands-On Exercise (2 hours)</b></summary>

**Exercise 1: Implement Game Controller (60 min)**

```python
class SnakeAndLadderGame:
    def __init__(self, board_size=100):
        self.board = Board(board_size)
        self.players = []
        self.current_player_index = 0
        self.dice_strategy = StandardDice()
        self.winner = None
        self.move_history = []
    
    def add_player(self, player_id, name):
        """TODO: Add player to game"""
        pass
    
    def set_dice_strategy(self, strategy):
        """TODO: Change dice strategy (Strategy Pattern!)"""
        pass
    
    def start_game(self):
        """TODO: Initialize game state"""
        pass
    
    def roll_dice_and_move(self):
        """
        TODO: Main game logic
        - Roll dice
        - Move current player
        - Check for winner
        - Switch to next player
        - Record move in history
        """
        pass
    
    def get_winner(self):
        """TODO: Return winner if game over"""
        pass
    
    def get_player_standings(self):
        """TODO: Return players sorted by position"""
        pass
    
    def get_statistics(self):
        """TODO: Return game statistics"""
        pass

# Test with controlled dice
game = SnakeAndLadderGame(board_size=20)
game.board.add_snake(15, 3)
game.board.add_ladder(5, 18)

# Use controlled dice for predictable testing
game.set_dice_strategy(ControlledDice([6, 4, 5]))

game.add_player("P1", "Alice")
game.add_player("P2", "Bob")
game.start_game()

# Simulate game
for i in range(10):
    move = game.roll_dice_and_move()
    print(f"{move['player']}: rolled {move['dice']}, "
          f"moved from {move['from']} to {move['to']}")
    
    if game.get_winner():
        print(f"Winner: {game.get_winner().name}")
        break
```

**Exercise 2: Add Game Variants (30 min)**

```python
class GameMode(Enum):
    CLASSIC = "classic"
    QUICK = "quick"  # Smaller board, fewer snakes/ladders
    HARDCORE = "hardcore"  # More snakes, fewer ladders

class GameFactory:
    @staticmethod
    def create_game(mode, num_players):
        """
        TODO: Create game with different configurations
        
        CLASSIC: 100 squares, 5 snakes, 5 ladders
        QUICK: 50 squares, 3 snakes, 3 ladders
        HARDCORE: 100 squares, 10 snakes, 2 ladders
        """
        pass

# Test
game = GameFactory.create_game(GameMode.QUICK, 2)
```

**Exercise 3: Implement Replay System (30 min)**

```python
class MoveRecord:
    def __init__(self, turn_number, player_id, dice_roll, 
                 from_pos, to_pos, event):
        self.turn_number = turn_number
        self.player_id = player_id
        self.dice_roll = dice_roll
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.event = event  # "NORMAL", "SNAKE", "LADDER", "WIN"

class GameReplay:
    def __init__(self, game):
        self.game = game
        self.moves = []
    
    def record_move(self, move_record):
        """TODO: Save move to history"""
        pass
    
    def replay(self):
        """TODO: Print all moves"""
        pass
    
    def get_key_moments(self):
        """
        TODO: Return interesting moments
        - Biggest snake hit
        - Longest ladder climb
        - Near misses (almost won)
        """
        pass
```

</details>

<details>
<summary><b>🎯 Design Questions</b></summary>

**Q1**: Why use Strategy pattern for dice instead of subclassing Game?  
**A**: Composition over inheritance. Can change dice behavior at runtime without changing game code.

**Q2**: How to prevent cheating (player modifying position directly)?  
**A**: Encapsulation: make position private (`self._position`), only modify through game controller

**Q3**: How to handle multiple winners (tie)?  
**A**: Track turn order, first player to reach wins, or allow simultaneous completion

**Q4**: How to add undo functionality?  
**A**: Store game state snapshots, or use Command pattern with undo() method

**Q5**: How to make game network-multiplayer?  
**A**: Separate game logic from I/O, use event-driven architecture, sync state over websockets

</details>

<details>
<summary><b>✅ Day 5 Checkpoint</b></summary>

- [ ] Implemented Strategy pattern correctly
- [ ] Game logic works with all dice strategies
- [ ] Statistics tracking functional
- [ ] Can test reliably with ControlledDice
- [ ] Understood turn-based game design

**Weekend**: Review Week 1, practice implementations

</details>

---

### Day 6-7: Week 1 Review & Practice

<details>
<summary><b>📋 Review Checklist</b></summary>

**Day 6: Deep Review**
- [ ] Re-implement Cache System (LRU) in 20 minutes
- [ ] Re-implement Parking Lot spot finder
- [ ] Re-implement Library fine calculator
- [ ] Write unit tests for all systems
- [ ] Identify your weakest area

**Day 7: Mock Interview**
- [ ] Record yourself solving Cache System (45 min)
- [ ] Watch recording, note improvements
- [ ] Practice explaining design decisions
- [ ] Prepare questions for Week 2

**Self-Assessment:**
Rate yourself 1-5 on:
- Data structures: ___/5
- OOP principles: ___/5
- Code cleanliness: ___/5
- Edge case handling: ___/5
- Communication: ___/5

**Week 1 Goal**: Score 4+ on all areas before Week 2

</details>

---

## Week 2: Intermediate Systems & Algorithms

### Day 8: Elevator System - SCAN Algorithm Deep Dive

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Understand and implement SCAN algorithm
- [ ] Compare FCFS, SSTF, SCAN, and LOOK algorithms
- [ ] Design multi-elevator dispatch system
- [ ] Handle concurrent requests efficiently
- [ ] Optimize for real-world scenarios (rush hour)

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**SCAN Algorithm** (Also called "Elevator Algorithm" or "Look Algorithm")

```python
"""
SCAN Algorithm - How Real Elevators Work

Concept:
1. Elevator continues in current direction until no more requests
2. Then reverses direction
3. Like a disk head scanning in operating systems

Example:
Current floor: 5, Direction: UP
Requests: [2, 3, 7, 9, 12, 14]
Up requests: [7, 9, 12, 14]
Down requests: [2, 3]

Movement: 5 -> 7 -> 9 -> 12 -> 14 (top) -> 3 -> 2 (serve down requests)

Why SCAN?
- Predictable wait times
- No starvation (everyone gets served eventually)
- Efficient for grouped requests
- Used in disk scheduling for same reasons
"""

class ElevatorController:
    def __init__(self, num_floors=10):
        self.current_floor = 0
        self.direction = "UP"  # UP, DOWN, IDLE
        self.up_requests = set()
        self.down_requests = set()
    
    def request(self, floor, direction):
        """Add request to appropriate set"""
        if direction == "UP":
            self.up_requests.add(floor)
        else:
            self.down_requests.add(floor)
    
    def next_floor(self):
        """
        SCAN logic: Continue in current direction
        """
        if self.direction == "UP":
            # Get next floor in up requests
            higher = [f for f in self.up_requests if f > self.current_floor]
            if higher:
                next_floor = min(higher)
                self.current_floor = next_floor
                self.up_requests.remove(next_floor)
                return next_floor
            else:
                # Reverse direction
                self.direction = "DOWN"
                return self.next_floor()
        
        elif self.direction == "DOWN":
            # Get next floor in down requests
            lower = [f for f in self.down_requests if f < self.current_floor]
            if lower:
                next_floor = max(lower)
                self.current_floor = next_floor
                self.down_requests.remove(next_floor)
                return next_floor
            else:
                # Reverse direction
                self.direction = "UP"
                return self.next_floor()
        
        return None  # No requests

# Trace through execution:
elevator = ElevatorController()
elevator.request(7, "UP")
elevator.request(3, "DOWN")
elevator.request(9, "UP")

print(f"Start: Floor {elevator.current_floor}, Direction: {elevator.direction}")
while elevator.up_requests or elevator.down_requests:
    next_floor = elevator.next_floor()
    print(f"Moving to floor {next_floor}, Direction: {elevator.direction}")
```

**Alternative Algorithms:**

1. **FCFS (First Come First Serve)**
   - Pros: Fair, simple
   - Cons: Inefficient, lots of back-and-forth

2. **SSTF (Shortest Seek Time First)**
   - Pros: Minimizes travel distance
   - Cons: Can cause starvation (far requests wait forever)

3. **LOOK (SCAN variant)**
   - Same as SCAN but doesn't go to top/bottom if no requests

</details>

<details>
<summary><b>✍️ Advanced Exercise: Multi-Elevator Dispatch</b></summary>

```python
class ElevatorDispatcher:
    """Assign request to best elevator"""
    
    def __init__(self, elevators):
        self.elevators = elevators
    
    def dispatch(self, request_floor, direction):
        """
        TODO: Implement dispatch algorithm
        
        Factors to consider:
        1. Distance from request floor
        2. Current direction (going towards request?)
        3. Current load (number of passengers)
        4. Number of stops before reaching request
        
        Scoring system:
        - Distance score: 20 - distance
        - Direction score: 10 if same direction, -5 if opposite
        - Load score: 20 - (current_load * 2)
        - Stop score: 10 - num_stops
        """
        best_elevator = None
        best_score = float('-inf')
        
        for elevator in self.elevators:
            score = self._calculate_score(elevator, request_floor, direction)
            if score > best_score:
                best_score = score
                best_elevator = elevator
        
        return best_elevator
    
    def _calculate_score(self, elevator, floor, direction):
        # TODO: Implement scoring logic
        pass

# Test with 4 elevators
elevators = [ElevatorController() for _ in range(4)]
elevators[0].current_floor = 5
elevators[1].current_floor = 10
elevators[2].current_floor = 1
elevators[3].current_floor = 8

dispatcher = ElevatorDispatcher(elevators)
best = dispatcher.dispatch(request_floor=7, direction="UP")
print(f"Assigned elevator at floor {best.current_floor}")
```

</details>

---

# 📚 PART 2: ADVANCED SYSTEMS (Week 3-4)

## Week 3: Distributed Systems Fundamentals

### Day 15: Rate Limiter - Distributed Algorithms

<details>
<summary><b>📖 Deep Dive: Token Bucket Algorithm</b></summary>

**Token Bucket Algorithm Explained**

```python
"""
Analogy: Think of a bucket that holds tokens

Rules:
1. Bucket has max capacity (burst limit)
2. Tokens added at constant rate (refill rate)
3. Each request consumes 1 token
4. If no tokens available, request is rejected
5. Tokens overflow when bucket is full

Why Token Bucket?
- Allows bursts (up to bucket capacity)
- Smooth rate limiting over time
- Industry standard (AWS, Stripe, etc.)

Parameters:
- capacity: Max tokens (burst size)
- refill_rate: Tokens added per second
- tokens: Current token count
- last_refill: Last refill timestamp
"""

import time
import threading

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        """
        capacity: Maximum tokens (burst size)
        refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def allow_request(self, tokens_needed=1):
        """
        Try to consume tokens. Return True if allowed.
        """
        with self.lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

# Example usage
limiter = TokenBucket(capacity=10, refill_rate=2)  # 2 tokens/sec

# Burst of 10 requests - all allowed
for i in range(10):
    print(f"Request {i+1}: {'✅ Allowed' if limiter.allow_request() else '❌ Blocked'}")

# 11th request - blocked (no tokens left)
print(f"Request 11: {'✅ Allowed' if limiter.allow_request() else '❌ Blocked'}")

# Wait 1 second - 2 tokens refilled
time.sleep(1)
print(f"After 1 sec: {'✅ Allowed' if limiter.allow_request() else '❌ Blocked'}")
```

**Comparison with Other Algorithms:**

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Token Bucket** | Allows bursts, smooth | More complex | APIs, most common |
| **Sliding Window** | Most accurate | Memory intensive | Critical operations |
| **Fixed Window** | Very simple, fast | Boundary issues | Non-critical rate limiting |
| **Leaky Bucket** | Smooth output | No bursts | Traffic shaping |

</details>

<details>
<summary><b>✍️ Exercise: Distributed Rate Limiter with Redis</b></summary>

```python
"""
Challenge: Implement rate limiter for distributed system

Problem:
- Multiple servers, each has local counter
- User hits Server A then Server B
- Each server allows N requests = 2N total (bypass!)

Solution: Centralized counter with Redis

Using Redis:
1. Atomic increment: INCR key
2. Set expiry: EXPIRE key seconds
3. Thread-safe across servers
"""

import redis
import time

class DistributedRateLimiter:
    def __init__(self, redis_client, max_requests, window_seconds):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds
    
    def allow_request(self, user_id):
        """
        TODO: Implement using Redis
        
        Approach 1: Simple Counter
        key = f"rate_limit:{user_id}"
        count = redis.incr(key)
        if count == 1:
            redis.expire(key, window_seconds)
        return count <= max_requests
        
        Approach 2: Sliding Window (more accurate)
        Use Redis Sorted Set with timestamps
        """
        pass
    
    def reset_limit(self, user_id):
        """Allow manual reset"""
        key = f"rate_limit:{user_id}"
        self.redis.delete(key)

# Test with Redis
r = redis.Redis(host='localhost', port=6379)
limiter = DistributedRateLimiter(r, max_requests=5, window_seconds=60)

# Simulate requests from different servers
for i in range(10):
    allowed = limiter.allow_request("user123")
    print(f"Request {i+1}: {'✅' if allowed else '❌'}")
```

**Production Considerations:**

1. **Redis Failure**: Fail open (allow all) or fail closed (block all)?
2. **Redis Replication**: Eventual consistency issues
3. **Network Latency**: Add timeout for Redis calls
4. **Cost**: Redis operations cost money at scale
5. **Monitoring**: Track rate limit hits, user patterns

</details>

<details>
<summary><b>🎯 System Design Question: API Rate Limiting</b></summary>

**Question**: "Design rate limiting for a REST API with 1M users"

**Approach:**

```
1. Requirements (5 min):
   - QPS (Queries Per Second): 100K
   - Rate limit: 1000 requests/hour per user
   - Global rate limit: 1M requests/min
   - Need analytics on rate limit violations

2. Components (10 min):
   ┌─────────────┐
   │   Client    │
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │ API Gateway │ ◄─ Rate Limiter Service
   └──────┬──────┘
          │
   ┌──────▼──────┐
   │   Backend   │
   └─────────────┘

3. Data Store Choice:
   - Redis (in-memory, fast, atomic operations)
   - Alternative: Cassandra (if Redis too expensive)

4. Algorithm:
   - Token Bucket for user-level (allow bursts)
   - Fixed Window for global limit (simpler, good enough)

5. Implementation:
   ```python
   class APIRateLimiter:
       def __init__(self):
           self.redis = redis.Redis()
           self.user_limiter = TokenBucket(...)
           self.global_limiter = FixedWindow(...)
       
       def check_rate_limit(self, user_id, endpoint):
           # Check global first (faster rejection)
           if not self.global_limiter.allow():
               return False, "Global rate limit exceeded"
           
           # Then check user-specific
           if not self.user_limiter.allow(user_id):
               self._log_violation(user_id, endpoint)
               return False, "User rate limit exceeded"
           
           return True, "OK"
   ```

6. Extensions:
   - Different limits per endpoint (/search: 10/sec, /user: 100/sec)
   - Premium users: higher limits
   - Distributed Redis cluster for HA
   - Rate limit response headers (X-RateLimit-Remaining)
```

</details>

---

### Day 16-17: Distributed Cache & KV Store

<details>
<summary><b>📖 Consistent Hashing Explained</b></summary>

**Why Consistent Hashing?**

```python
"""
Problem with Simple Hashing:
- server_id = hash(key) % num_servers
- Add/remove server → rehash ALL keys!
- If you have 1M keys and add 1 server, 999,999 keys move

Consistent Hashing Solution:
- Hash servers onto a ring (0 to 2^32-1)
- Hash keys onto same ring
- Key goes to next server clockwise
- Add/remove server → only K/N keys move (K = total keys, N = servers)

Virtual Nodes:
- Each physical server → 150 virtual nodes
- Better load distribution
- When server fails, load spreads across multiple servers
"""

import hashlib
import bisect

class ConsistentHashRing:
    def __init__(self, num_virtual_nodes=150):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring = []  # Sorted list of hash values
        self.hash_to_node = {}  # hash -> node_id
    
    def _hash(self, key):
        """MD5 hash to integer"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id):
        """Add node with virtual nodes"""
        for i in range(self.num_virtual_nodes):
            virtual_key = f"{node_id}:vnode{i}"
            hash_val = self._hash(virtual_key)
            
            # Insert in sorted order
            bisect.insort(self.ring, hash_val)
            self.hash_to_node[hash_val] = node_id
    
    def remove_node(self, node_id):
        """Remove all virtual nodes for this physical node"""
        to_remove = []
        for hash_val, node in self.hash_to_node.items():
            if node == node_id:
                to_remove.append(hash_val)
        
        for hash_val in to_remove:
            self.ring.remove(hash_val)
            del self.hash_to_node[hash_val]
    
    def get_node(self, key):
        """Find which node should store this key"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # Binary search for next node clockwise
        idx = bisect.bisect_right(self.ring, hash_val)
        if idx == len(self.ring):
            idx = 0  # Wrap around
        
        return self.hash_to_node[self.ring[idx]]

# Demo
ring = ConsistentHashRing()
ring.add_node("server1")
ring.add_node("server2")
ring.add_node("server3")

# Check key distribution
keys = [f"user{i}" for i in range(1000)]
distribution = {}
for key in keys:
    node = ring.get_node(key)
    distribution[node] = distribution.get(node, 0) + 1

print("Key distribution:")
for node, count in distribution.items():
    print(f"  {node}: {count} keys ({count/10:.1f}%)")

# Remove a node - see how few keys move
print("\nRemoving server2...")
ring.remove_node("server2")

moved = 0
for key in keys:
    new_node = ring.get_node(key)
    # In practice, you'd compare with old_node stored somewhere
    # For demo, we know ~1/3 should move

print(f"Approximately {len(keys)//3} keys moved")
```

</details>

<details>
<summary><b>✍️ Exercise: Vector Clocks for Conflict Detection</b></summary>

```python
"""
Vector Clocks: Track causality in distributed systems

Problem:
- Server A and Server B both update same key
- Which update happened first?
- Timestamps don't work (clock skew)

Solution: Vector Clocks
- Each node has a clock for every node
- [A:1, B:0, C:0] means A performed 1 operation
- [A:1, B:1, C:0] means A did 1, B did 1

Comparison:
- VC1 < VC2 if VC1[i] <= VC2[i] for all i (VC1 "happens-before" VC2)
- VC1 || VC2 if neither < nor > (concurrent updates!)
"""

class VectorClock:
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.clock = {f"node{i}": 0 for i in range(num_nodes)}
    
    def increment(self):
        """Increment this node's counter"""
        self.clock[self.node_id] += 1
    
    def update(self, other_clock):
        """Merge with another vector clock"""
        for node, count in other_clock.items():
            self.clock[node] = max(self.clock[node], count)
        self.increment()  # Then increment own counter
    
    def happens_before(self, other):
        """Check if this clock happens-before other"""
        # TODO: Implement comparison logic
        # Return True if self <= other for all components
        # and self < other for at least one component
        pass
    
    def concurrent(self, other):
        """Check if concurrent (neither happens-before the other)"""
        return not self.happens_before(other) and not other.happens_before(self)

# Test scenario
vc_a = VectorClock("nodeA", 3)
vc_b = VectorClock("nodeB", 3)
vc_c = VectorClock("nodeC", 3)

# Events
vc_a.increment()  # A: [A:1, B:0, C:0]
vc_b.increment()  # B: [A:0, B:1, C:0]
# A and B are concurrent!

vc_a.update(vc_b.clock)  # A: [A:2, B:1, C:0]
# Now A happens-after B

print(f"A happens-before C: {vc_a.happens_before(vc_c)}")
print(f"A concurrent with B (original): {concurrent_check}")
```

</details>

---

## Week 4: Polish & Practice

### Day 22-28: Mock Interviews & Review

<details>
<summary><b>📋 Mock Interview Schedule</b></summary>

**Day 22**: Mock Interview #1 - Cache System
- Record yourself with phone/webcam
- 45 minutes strict timing
- Review recording, note mistakes

**Day 23**: Mock Interview #2 - Rate Limiter
- Practice with a friend/colleague
- Get feedback on communication

**Day 24**: Mock Interview #3 - Random System
- Use random number generator to pick system
- No preparation time
- Tests true understanding

**Day 25**: Review weak areas
- Identify patterns in mistakes
- Re-implement systems you struggled with

**Day 26**: Behavioral prep
- Prepare STAR stories for each system
- "Tell me about a time you optimized performance"
- Use your implementations as examples

**Day 27**: System design integration
- Connect LLD to HLD
- "How does your cache fit into overall architecture?"

**Day 28**: Final review & confidence building
- Skim all systems
- Review cheat sheet
- Get good sleep!

</details>

---

# 🎓 APPENDIX

## Appendix A: Quick Reference Tables

### Data Structure Selection Guide

| Need | Use This | Time | Space |
|------|----------|------|-------|
| Fast lookup by key | HashMap | O(1) avg | O(n) |
| Maintain order + fast lookup | TreeMap | O(log n) | O(n) |
| Recently used items | LinkedHashMap | O(1) | O(n) |
| Priority queue | Heap | O(log n) | O(n) |
| Prefix search | Trie | O(m) m=length | O(ALPHABET_SIZE * n) |
| Range queries | Segment Tree | O(log n) | O(n) |
| Fast insertion/deletion at ends | Deque | O(1) | O(n) |

### Design Pattern Decision Matrix

| Situation | Pattern | When NOT to Use |
|-----------|---------|-----------------|
| Multiple algorithms | Strategy | Only 1-2 variants |
| Complex object creation | Builder/Factory | Simple constructors |
| One instance globally | Singleton | Can use DI instead |
| Subscribe to events | Observer | One-to-one relationships |
| State transitions | State | Few simple states |
| Add functionality | Decorator | Use inheritance |
| Simplify interface | Facade | Interface already simple |

---

## Appendix B: Debugging Checklist

When stuck on a bug during interview:

- [ ] **Off-by-one errors**: Check array indices, loop conditions
- [ ] **Null/None checks**: Always validate input
- [ ] **Edge cases**: Empty input, single element, all same values
- [ ] **Integer overflow**: Use appropriate data types
- [ ] **Concurrent access**: Check thread safety
- [ ] **Memory leaks**: Ensure cleanup (close files, remove references)
- [ ] **Time complexity**: Is your solution actually O(n)?

**Debugging Process:**
1. **Trace with example**: Walk through code with sample input
2. **Print statements**: Add logs at critical points
3. **Rubber duck**: Explain code line-by-line out loud
4. **Binary search**: Comment out half, find which half has bug
5. **Ask for help**: "I think the issue is here, but not sure why..."

---

## Appendix C: Resources by Learning Style

### Visual Learners 📊
- Draw diagrams for data structures
- Watch YouTube videos (Gaurav Sen, Tech Dummies)
- Create flowcharts for algorithms
- Use visualization tools (visualgo.net)

### Auditory Learners 🎧
- Record yourself explaining concepts
- Listen to system design podcasts
- Join study groups
- Explain to others (teach = learn)

### Kinesthetic Learners ✍️
- Type out code (don't copy-paste)
- Build projects using patterns
- Debug by modifying code
- Practice on whiteboard/paper

---

## Appendix D: Company-Specific Tips

### Amazon Leadership Principles Applied to LLD

1. **Customer Obsession**
   - "I designed the cache with <1ms latency for better user experience"

2. **Ownership**
   - "I'd add monitoring and alerts to track cache hit rate"

3. **Invent and Simplify**
   - "Instead of complex B-tree, HashMap + Linked List achieves O(1)"

4. **Bias for Action**
   - "Let me start with basic implementation, then optimize"

5. **Frugality**
   - "Using LRU to stay within memory limits"

### Google-Style Interview Tips

- Focus on **scalability** ("how would this work with 1B users?")
- Discuss **trade-offs** extensively
- Consider **global distribution**
- Mention **SRE concerns** (monitoring, SLAs)

### Meta-Style Interview Tips

- Emphasize **social features** (notifications, feeds)
- Discuss **real-time updates**
- Consider **mobile** constraints
- Focus on **user engagement** metrics

---

## Appendix E: Progress Tracker

### Week 1 Checklist
- [ ] Day 1: Data structures & OOP
- [ ] Day 2: Cache System
- [ ] Day 3: Parking Lot
- [ ] Day 4: Library Management
- [ ] Day 5: Snake & Ladder
- [ ] Day 6: Review & practice
- [ ] Day 7: Mock interview #1

**Self-Rating** (1-5): _____

**Confidence Level**: 😟 😐 🙂 😃 😎

**Notes**: ________________________________

### Week 2 Checklist
- [ ] Day 8: Elevator System
- [ ] Day 9: Vending Machine
- [ ] Day 10: URL Shortener
- [ ] Day 11: Chess Game
- [ ] Day 12: Autocomplete
- [ ] Day 13: Review algorithms
- [ ] Day 14: Mock interview #2

**Self-Rating** (1-5): _____

### Week 3 Checklist
- [ ] Day 15: Rate Limiter
- [ ] Day 16: Job Processor
- [ ] Day 17: Distributed Cache
- [ ] Day 18: KV Store
- [ ] Day 19: File Storage
- [ ] Day 20: Notification Service
- [ ] Day 21: Mock interview #3

**Self-Rating** (1-5): _____

### Week 4 Checklist
- [ ] Day 22-28: Intensive practice
- [ ] Final mock interviews
- [ ] Review all systems
- [ ] Ready for real interviews!

---

## 📞 Final Encouragement

> **"Every expert was once a beginner. Every master was once a student."**

You've got this! Remember:

1. **Consistency > Intensity**: 1 hour daily beats 7 hours once/week
2. **Understanding > Memorization**: Focus on "why" not just "how"
3. **Practice > Theory**: Code it, don't just read it
4. **Progress > Perfection**: Done is better than perfect

**Good luck with your interviews! 🚀**

---

<div align="center">

*Study Guide Version 2.0 | December 2025*

**Questions? Issues? Suggestions?**  
Open an issue on GitHub or contribute improvements!

[⬆️ Back to Top](#-complete-lld-interview-study-guide)

</div>
