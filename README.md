# 🎯 Low Level Design (LLD) Interview Mastery

<div align="center">

**16 Production-Ready System Implementations | Comprehensive Interview Guides | Best Practices**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Systems](https://img.shields.io/badge/Systems-16-orange.svg)](#-complete-system-catalog)
[![Lines of Code](https://img.shields.io/badge/Lines-6000+-purple.svg)](#)
[![Study Hours](https://img.shields.io/badge/Study%20Hours-80+-red.svg)](#-interview-preparation-roadmap)

*Master Low Level Design interviews with battle-tested implementations and expert guidance*

**[🚀 Quick Start](#-quick-start)** • 
**[📖 Systems Catalog](#-complete-system-catalog)** • 
**[🗺️ Study Roadmap](#-interview-preparation-roadmap)** • 
**[⚡ Cheat Sheet](#-quick-reference-cheat-sheet)** • 
**[🎭 Interview Tips](#-ace-your-lld-interview---expert-tips)**

</div>

---

## 📑 Table of Contents

<details open>
<summary><b>Click to expand/collapse</b></summary>

### Getting Started
- [📚 What You'll Learn](#-what-youll-learn)
- [🚀 Quick Start](#-quick-start)
- [⚡ Quick Reference Cheat Sheet](#-quick-reference-cheat-sheet)

### System Catalog & Learning Path
- [📋 Complete System Catalog](#-complete-system-catalog) - All 16 systems organized by category
- [🧭 System Selection Guide](#-system-selection-guide) - Which system to study first?
- [📊 System Comparison Matrix](#-system-comparison-matrix) - Difficulty, frequency, time investment

### Detailed Systems (Click to jump)
- [Caching & Storage](#️-by-system-design-category) - Cache, Distributed Cache, KV Store, File Storage
- [Concurrency](#️-by-system-design-category) - Rate Limiter, Job Processor
- [Infrastructure](#️-by-system-design-category) - URL Shortener, Notification Service
- [Domain-Specific](#️-by-system-design-category) - Elevator, Vending Machine, Library, Parking
- [Games](#️-by-system-design-category) - Chess, Snake & Ladder
- [File & Search](#️-by-system-design-category) - File System, Autocomplete

### Interview Preparation
- [🗺️ 4-Week Study Plan](#-interview-preparation-roadmap) - Week-by-week breakdown
- [🎭 Interview Tips & Strategies](#-ace-your-lld-interview---expert-tips)
- [📊 Complexity Comparison](#-complexity-comparison) - Time/space for all operations
- [🎤 Common Interview Questions](#-common-interview-questions-across-all-systems)

### Additional Resources
- [🚀 Beyond the Basics](#-beyond-the-basics---advanced-topics)
- [📚 Recommended Resources](#-recommended-resources) - Books, courses, videos
- [🤝 Contributing](#-contributing--community)

</details>

---

## 📚 What You'll Learn

This repository provides **16 production-grade system implementations** covering the most common LLD interview problems. Each system includes:

✅ **Complete working code** with comprehensive demos  
✅ **Detailed interview guides** with Q&A sections  
✅ **Design pattern explanations** (Strategy, Factory, Observer, etc.)  
✅ **Complexity analysis** for all key operations  
✅ **Real-world considerations** (scalability, fault tolerance, monitoring)  
✅ **Thread safety** and concurrency patterns  

**Perfect for:** Software Engineers preparing for interviews at FAANG, startups, and product companies.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/iamsuryansh/lld_practice.git
cd lld_practice

# No dependencies needed - Python 3.8+ standard library only!

# Run any system (they all have demo scenarios)
python3 01_cache_system.py
python3 13_distributed_cache.py
python3 15_autocomplete_system.py
```

**Each system runs 4-6 comprehensive demos** showing core functionality, edge cases, and production scenarios.

---

## ⚡ Quick Reference Cheat Sheet

### 🎯 Interview Day Emergency Guide

<table>
<tr>
<td width="50%">

**📋 30-Second Checklist**
- [ ] Clarify requirements (5 min)
- [ ] Draw high-level diagram
- [ ] Identify core classes
- [ ] Define key methods
- [ ] Start with simplest version
- [ ] Add complexity incrementally
- [ ] Discuss trade-offs
- [ ] Mention production concerns

</td>
<td width="50%">

**🔑 Magic Questions to Ask**
- "What's the expected scale?"
- "Any specific constraints?"
- "Read-heavy or write-heavy?"
- "Need persistence?"
- "Thread safety required?"
- "Latency requirements?"
- "Consistency vs availability?"

</td>
</tr>
</table>

### 🏗️ Common Design Patterns - When to Use

| Pattern | Use When | Example System | Code Hint |
|---------|----------|----------------|-----------|
| **Strategy** | Multiple algorithms, runtime selection | Cache (LRU/LFU/FIFO) | `interface Strategy { execute() }` |
| **Factory** | Complex object creation | Rate Limiter types | `static create(type)` method |
| **Singleton** | Single instance needed | Parking Lot controller | `private constructor + getInstance()` |
| **Observer** | Event notifications | Library (book returns) | `notify(subscribers)` |
| **State** | State transitions | Vending Machine | `currentState.handle(request)` |
| **Builder** | Many parameters | System configurations | `Builder().with().build()` |
| **Adapter** | Interface compatibility | Payment processors | `wrap(external).toInternal()` |

### 🚀 Data Structure Decision Tree

```
Need O(1) access by key? → HashMap/Dictionary
Need ordering? → TreeMap / Sorted structures
Need recently used tracking? → LinkedHashMap + pointers
Need frequency tracking? → HashMap + Counter/Heap
Need prefix search? → Trie
Need range queries? → Segment Tree / Interval Tree
Need thread-safe? → ConcurrentHashMap / Locks
```

### 📊 Complexity Quick Reference

| Operation | Target | Acceptable | Red Flag |
|-----------|--------|------------|----------|
| Cache get/put | O(1) | O(log n) | O(n) |
| Rate limit check | O(1) | O(log n) | O(n) |
| Find parking spot | O(1) with index | O(n) linear search | O(n²) |
| Shorten URL | O(1) avg | O(log n) | O(n) |
| Autocomplete | O(m + k) | O(m + n) | O(n²) |

*m = query length, n = total words, k = results*

---

## 📋 Complete System Catalog

### 🎓 Difficulty Levels

| Difficulty | Systems | Focus Area |
|------------|---------|------------|
| 🟢 **Beginner** | Cache, Parking Lot, Library, Snake & Ladder | Core OOP, data structures |
| 🟡 **Intermediate** | Elevator, Vending Machine, URL Shortener, Chess | Algorithms, state machines |
| 🔴 **Advanced** | Rate Limiter, Job Processor, Distributed Cache, KV Store | Distributed systems, concurrency |

---

### 🏗️ By System Design Category

#### **Caching & Storage Systems**
- [#1 Advanced Cache System](#1-advanced-cache-system-) - LRU/LFU/FIFO eviction policies
- [#13 Distributed Cache](#13-distributed-cache-system-) - Consistent hashing, quorum consensus
- [#14 Distributed KV Store](#14-distributed-key-value-store-️) - Vector clocks, Merkle trees, WAL
- [#16 File Storage System](#16-file-storage-system-) - Chunking, deduplication, delta sync

#### **Concurrency & Rate Limiting**
- [#2 Rate Limiter](#2-rate-limiter-system-️) - Token bucket, sliding window algorithms
- [#3 Distributed Job Processor](#3-distributed-job-processor-️) - Thread pools, retry mechanisms

#### **Infrastructure & APIs**
- [#8 URL Shortener](#8-url-shortener-) - Base62 encoding, collision handling
- [#10 Notification Service](#10-notification-service-) - Multi-channel delivery, deduplication

#### **Domain-Specific Systems**
- [#4 Elevator System](#4-elevator-system-) - SCAN algorithm, dispatch optimization
- [#5 Vending Machine](#5-vending-machine-) - State pattern, payment processing
- [#6 Library Management](#6-library-management-system-) - Business logic, fine calculation
- [#7 Parking Lot](#7-parking-lot-system-️) - Spot allocation, pricing tiers

#### **Games & Interactive Systems**
- [#9 Chess Game](#9-chess-game-️) - Move validation, check/checkmate detection
- [#12 Snake & Ladder Game](#12-snake-and-ladder-game-) - Strategy pattern, turn management

#### **File & Search Systems**
- [#11 File System](#11-file-system-) - Path resolution, permissions, hierarchical structure
- [#15 Autocomplete System](#15-autocomplete-system-) - Trie, fuzzy matching, caching

---

## 🧭 System Selection Guide

**"Which system should I study first?"** - Use this decision tree:

```
START HERE
    │
    ├─ New to LLD? → Start with #1 Cache System (core data structures)
    │                 Then → #7 Parking Lot (basic OOP)
    │
    ├─ Preparing for specific company?
    │   ├─ FAANG → Focus on #2 Rate Limiter, #13 Distributed Cache, #14 KV Store
    │   ├─ Startup → Focus on #8 URL Shortener, #3 Job Processor, #10 Notification
    │   └─ E-commerce → Focus on #5 Vending Machine, #6 Library, #7 Parking Lot
    │
    ├─ Want to learn distributed systems? → #13, #14, #2, #3 (in order)
    │
    ├─ Weak on algorithms? → #4 Elevator (SCAN), #15 Autocomplete (Trie)
    │
    └─ Interview in <1 week? → Study #1, #2, #7, #8 (most common questions)
```

### 📊 System Comparison Matrix

| System | Difficulty | Time to Master | Interview Frequency | Key Learning |
|--------|------------|----------------|---------------------|--------------|
| Cache (#1) | 🟢 Easy | 2-3 hours | ⭐⭐⭐⭐⭐ Very High | HashMap + LinkedList |
| Rate Limiter (#2) | 🔴 Hard | 4-5 hours | ⭐⭐⭐⭐⭐ Very High | Distributed algorithms |
| Job Processor (#3) | 🔴 Hard | 5-6 hours | ⭐⭐⭐⭐ High | Concurrency patterns |
| Elevator (#4) | 🟡 Medium | 3-4 hours | ⭐⭐⭐ Medium | SCAN algorithm |
| Vending Machine (#5) | 🟡 Medium | 2-3 hours | ⭐⭐ Low | State pattern |
| Library (#6) | 🟢 Easy | 2-3 hours | ⭐⭐ Low | Business logic |
| Parking Lot (#7) | 🟢 Easy | 2-3 hours | ⭐⭐⭐⭐ High | OOP basics |
| URL Shortener (#8) | 🟡 Medium | 3-4 hours | ⭐⭐⭐⭐⭐ Very High | Encoding schemes |
| Chess (#9) | 🔴 Hard | 6-8 hours | ⭐⭐ Low | Complex rules |
| Notification (#10) | 🟡 Medium | 3-4 hours | ⭐⭐⭐ Medium | Multi-channel routing |
| File System (#11) | 🟡 Medium | 3-4 hours | ⭐⭐⭐ Medium | Tree structures |
| Snake & Ladder (#12) | 🟢 Easy | 2 hours | ⭐⭐ Low | Strategy pattern |
| Distributed Cache (#13) | 🔴 Hard | 6-8 hours | ⭐⭐⭐⭐⭐ Very High | Consistent hashing |
| KV Store (#14) | 🔴 Hard | 8-10 hours | ⭐⭐⭐⭐ High | Vector clocks |
| Autocomplete (#15) | 🟡 Medium | 3-4 hours | ⭐⭐⭐⭐ High | Trie structure |
| File Storage (#16) | 🔴 Hard | 6-8 hours | ⭐⭐⭐ Medium | Chunking & dedup |

**Total study time**: 60-80 hours for complete mastery of all 16 systems

---

## 🎯 Systems Included

### 1. [Advanced Cache System](./01_cache_system_readme.md) 📦
**File**: `01_cache_system.py`

**What it covers:**
- **Cache eviction policies**: LRU, LFU, FIFO
- **TTL (Time-to-Live)**: Automatic expiration
- **Thread safety**: Concurrent access handling
- **Design patterns**: Strategy, Factory

**Interview focus:**
- Data structure selection (HashMap + Linked List)
- Algorithm complexity analysis
- Thread safety considerations
- Real-world scalability challenges

**Key algorithms:**
- LRU: O(1) get/put using HashMap + Doubly Linked List
- LFU: O(1) get/put using HashMap + Frequency tracking
- FIFO: O(1) get/put using HashMap + Queue

---

### 2. [Rate Limiter System](./02_rate_limiter_readme.md) ⏱️
**File**: `02_rate_limiter.py`

**What it covers:**
- **Rate limiting algorithms**: Token Bucket, Sliding Window, Fixed Window
- **Distributed systems**: Multi-server coordination
- **Performance trade-offs**: Accuracy vs Speed vs Memory
- **Design patterns**: Strategy, Factory

**Interview focus:**
- Algorithm selection based on requirements
- Distributed system challenges
- Storage backend considerations (Redis, Database)
- Scalability and fault tolerance

**Key algorithms:**
- Token Bucket: Burst-friendly, industry standard
- Sliding Window: Most accurate, memory intensive
- Fixed Window: Simplest, boundary issues

---

### 3. [Distributed Job Processor](./03_job_processor_readme.md) ⚙️
**File**: `03_job_processor.py`

**What it covers:**
- **Job scheduling**: FIFO, Priority-based, Delayed execution
- **Concurrency**: Worker thread pools, job isolation
- **Reliability**: Retry mechanisms, exponential backoff
- **System design**: Queue architectures, fault tolerance

**Interview focus:**
- Concurrency and thread safety
- Distributed processing coordination  
- Reliability and error handling
- Job orchestration and dependencies

**Key concepts:**
- Worker pool management
- Job state tracking and persistence
- Retry policies and dead letter queues
- Resource management and isolation

---

### 4. [Elevator System](./04_elevator_system_readme.md) 🛗
**File**: `04_elevator_system.py`

**What it covers:**
- **SCAN Algorithm**: Efficient elevator scheduling (from disk scheduling)
- **Dispatch strategies**: Nearest car, optimized, zone-based
- **Load balancing**: Multi-elevator coordination
- **Design patterns**: Strategy, Observer

**Interview focus:**
- Algorithm selection (SCAN vs FCFS vs SSTF)
- State machine design
- Concurrent request handling
- Peak load optimization (rush hours)

**Key algorithms:**
- SCAN: O(1) movement, O(log n) request insertion
- Dispatch: O(m) where m = number of elevators
- Score-based selection considering distance, direction, load

---

### 5. [Vending Machine](./05_vending_machine_readme.md) 🏪
**File**: `05_vending_machine.py`

**What it covers:**
- **State Pattern**: IDLE → PAYMENT_RECEIVED → DISPENSING → ERROR
- **Payment Processing**: Cash, Card, Digital wallet strategies
- **Change calculation**: Greedy algorithm for canonical coin systems
- **Error handling**: Mechanical failures, refunds, recovery

**Interview focus:**
- State machine design and transitions
- Payment processing strategies
- Money calculation (avoiding float errors)
- Inventory management and audit trails

**Key concepts:**
- Work in cents to avoid floating point errors
- Strategy pattern for multiple payment types
- Comprehensive transaction logging
- Concurrent access with RLock

---

### 6. [Library Management System](./06_library_management_readme.md) 📚
**File**: `06_library_management.py`

**What it covers:**
- **Catalog management**: Books, authors, genres, ISBN
- **Membership system**: Different member types, borrowing limits
- **Fine calculation**: Overdue fees, late return penalties
- **Search functionality**: By title, author, ISBN, genre

**Interview focus:**
- Complex entity relationships
- Business rule enforcement (borrowing limits, fines)
- Search and indexing strategies
- Reservation and waitlist management

**Key concepts:**
- One-to-many and many-to-many relationships
- Date handling and due date calculations
- Notification systems (reminders, overdue alerts)
- Report generation (popular books, revenue)

---

### 7. [Parking Lot System](./07_parking_lot_readme.md) 🅿️
**File**: `07_parking_lot.py`

**What it covers:**
- **Spot allocation**: Different vehicle types (compact, large, motorcycle)
- **Pricing strategies**: Hourly, daily, flat rate
- **Payment processing**: Entry/exit ticket validation
- **Capacity management**: Floor-wise, type-wise availability

**Interview focus:**
- Space optimization algorithms
- Real-time availability tracking
- Pricing calculation with multiple rules
- Concurrency (multiple vehicles entering/exiting)

**Key algorithms:**
- Nearest spot allocation: O(n) worst case, O(1) with indexing
- Pricing calculation: Strategy pattern
- Capacity tracking: O(1) with counters

---

### 8. [URL Shortener](./08_url_shortener_readme.md) 🔗
**File**: `08_url_shortener.py`

**What it covers:**
- **Encoding schemes**: Base62 encoding for short URLs
- **Collision handling**: Hash-based, counter-based approaches
- **Analytics**: Click tracking, geographic data, referrers
- **Expiration**: TTL for temporary short URLs

**Interview focus:**
- Hash function design
- Database schema for billion-scale URLs
- Distributed counter generation
- Cache strategy for hot URLs

**Key algorithms:**
- Base62 encoding: 62^7 ≈ 3.5 trillion unique URLs
- Hash collisions: Linear probing or chaining
- Analytics: Time-series data storage

---

### 9. [Chess Game](./09_chess_game_readme.md) ♟️
**File**: `09_chess_game.py`

**What it covers:**
- **Board representation**: 8x8 grid with position tracking
- **Piece movement**: All 6 piece types with validation
- **Special moves**: Castling, En Passant, Pawn Promotion
- **Game state detection**: Check, Checkmate, Stalemate

**Interview focus:**
- Object-oriented design with inheritance
- Strategy pattern for piece-specific logic
- Move validation and simulation
- Complex rule enforcement

**Key algorithms:**
- Move validation: O(n²) for check detection
- Piece movement: O(1) to O(n) depending on piece type
- Game state detection: O(n³) for legal move enumeration

---

### 10. [Notification Service](./10_notification_service_readme.md) 📧
**File**: `10_notification_service.py`

**What it covers:**
- **Multi-channel delivery**: Email, SMS, Push, Slack notifications
- **Deduplication**: Time-window based spam prevention
- **Rate limiting**: Sliding window algorithm per user/channel
- **Retry mechanisms**: Exponential backoff for failed deliveries
- **Priority queue**: Critical notifications delivered first

**Interview focus:**
- Strategy pattern for notification channels
- Distributed deduplication with hash-based detection
- Rate limiting algorithms (sliding window vs token bucket)
- Reliability patterns (retry, backoff, dead letter queue)
- Queue management and priority scheduling

**Key algorithms:**
- Deduplication: O(1) hash lookup with SHA-256
- Rate limiting: O(log n) sliding window with timestamp cleanup
- Priority queue: O(log n) insertion, O(1) peek
- Retry backoff: Exponential delay calculation O(1)

---

### 11. [File System](./11_file_system_readme.md) 📁
**File**: `11_file_system.py`

**What it covers:**
- **Hierarchical structure**: Tree-based directory organization
- **File operations**: Create, delete, rename, move, read, write
- **Unix-style permissions**: Read, write, execute for owner/group/others
- **Path resolution**: Absolute and relative path parsing with normalization
- **Tree traversal**: DFS and BFS algorithms
- **Search functionality**: Find files by name, size, type, modification time

**Interview focus:**
- Composite pattern for uniform file/directory interface
- Tree data structure and navigation algorithms
- Permission bit manipulation and checking
- Path parsing and normalization (handling ., .., etc.)
- Cycle detection for move operations
- Cache invalidation strategies

**Key algorithms:**
- Path normalization: O(n) using stack for resolving . and ..
- Path resolution: O(depth) tree traversal, O(1) with caching
- Tree traversal: O(n) for both DFS and BFS
- Move with cycle detection: O(depth) ancestor checking

---

### 12. [Snake and Ladder Game](./12_snake_ladder_game_readme.md) 🎲
**File**: `12_snake_ladder_game.py`

**What it covers:**
- **Classic board game**: 100-cell board with snakes and ladders
- **Multiplayer support**: Turn-based gameplay with 2+ players
- **Strategy pattern**: Flexible dice rolling (standard, weighted, controlled)
- **Game state management**: Move history, statistics tracking
- **Thread safety**: Concurrent multiplayer access with RLock
- **Testing support**: Controlled dice for deterministic unit tests

**Interview focus:**
- Strategy pattern for algorithm flexibility (dice rolling)
- Dictionary-based sparse data structure (O(1) lookups)
- Turn rotation and player state management
- Boundary handling (exact roll to win)
- Statistics tracking and audit trails
- Thread safety in multiplayer scenarios

**Key algorithms:**
- Position lookup: O(1) using HashMap for snakes/ladders
- Dice roll: O(1) for all strategies
- Turn rotation: O(1) with modulo arithmetic
- Move validation: O(1) boundary checks
- Statistics: O(P log P) for player standings (P = players)

---

### 13. [Distributed Cache System](./13_distributed_cache_readme.md) 🌐
**File**: `13_distributed_cache.py`

**What it covers:**
- **Consistent hashing**: Data partitioning with virtual nodes for load balancing
- **Replication**: Configurable replication factor (default: 3) for fault tolerance
- **Consistency levels**: ONE, QUORUM, ALL for CAP theorem trade-offs
- **Read repair**: Automatic stale replica detection and correction
- **Hot key detection**: Identify and mitigate skewed workloads
- **Node failure handling**: Automatic failover with minimal data movement
- **TTL expiration**: Per-entry time-to-live with lazy eviction

**Interview focus:**
- Why consistent hashing over modulo (minimal remapping: K/N vs all keys)
- Quorum math: W + R > N for consistency guarantees
- Read repair vs anti-entropy (lazy vs eager consistency)
- Hot key mitigation strategies (local caching, key splitting, increased replication)
- CAP theorem trade-offs (CP vs AP modes)
- Virtual nodes for heterogeneous cluster load balancing

**Key algorithms:**
- Consistent hashing: O(log N) node lookup with binary search
- Add/remove node: O(V log N) where V = virtual nodes per physical node
- Quorum read: O(R) RPCs where R = read quorum
- Quorum write: O(W) RPCs where W = write quorum
- Read repair: O(R) background updates to stale replicas

---

### 14. Distributed Key-Value Store 🗄️
**File**: `14_distributed_kv_store.py`

**What it covers:**
- **Vector clocks**: Causality tracking for conflict detection
- **Merkle trees**: Efficient anti-entropy synchronization (O(log N))
- **Write-ahead log (WAL)**: Durability and crash recovery
- **Multi-version concurrency control (MVCC)**: Keep multiple versions per key
- **Quorum consensus**: Tunable consistency (R + W > N)
- **Consistent hashing**: Data partitioning across nodes

**Interview focus:**
- Vector clocks vs timestamps (causality vs wall-clock time)
- Merkle tree benefits (O(log N) sync vs O(N) full comparison)
- WAL replay algorithm for crash recovery
- Conflict resolution with concurrent writes
- CAP theorem trade-offs (tunable R, W, N)
- Comparison with Dynamo, Cassandra, Riak

**Key algorithms:**
- Vector clock operations: O(N) where N = number of nodes
- Merkle tree build: O(M log M) where M = keys
- Merkle tree compare: O(log M) to find divergence
- WAL replay: O(entries) with sequential reads
- Quorum get/put: O(R) or O(W) RPCs

---

### 15. Autocomplete System 🔍
**File**: `15_autocomplete_system.py`

**What it covers:**
- **Trie data structure**: Efficient prefix search
- **Frequency-based ranking**: Popular suggestions first
- **LRU caching**: Cache popular prefixes for <1ms latency
- **Fuzzy matching**: Levenshtein distance for typo correction
- **Top-K heap**: Get K most relevant suggestions efficiently

**Interview focus:**
- Trie vs other data structures (HashMap, Array)
- Space-time trade-offs (Trie size vs search speed)
- Caching strategy for real-time performance
- Fuzzy matching algorithms and complexity
- Personalization (user history, location)
- Scalability (billions of queries per day)

**Key algorithms:**
- Trie insert/search: O(M) where M = word length
- Get suggestions: O(M + N) where N = matching words
- Top-K with heap: O(N log K) for K results
- Levenshtein distance: O(M × N) dynamic programming
- LRU cache: O(1) get/put operations

---

### 16. File Storage System 📁
**File**: `16_file_storage_system.py`

**What it covers:**
- **Content-addressable storage**: SHA-256 chunk IDs for deduplication
- **File chunking**: Fixed or variable-size with rolling hash
- **Delta sync**: Transfer only changed chunks (rsync algorithm)
- **Version control**: Immutable snapshots with parent pointers
- **Conflict resolution**: Last-write-wins and 3-way merge
- **Reference counting**: Automatic chunk garbage collection

**Interview focus:**
- Chunking strategies (fixed vs variable, Rabin fingerprinting)
- Deduplication benefits (40-60% space savings)
- Delta sync algorithm (rsync-style, 90% bandwidth savings)
- Version control design (Git-like snapshots)
- Conflict detection and resolution strategies
- Production deployment (S3, CDN, encryption)

**Key algorithms:**
- SHA-256 hashing: O(N) where N = file size
- Fixed-size chunking: O(N / chunk_size)
- Delta sync: O(N) to compute, O(changed) to transfer
- Three-way merge: O(chunks) comparison
- Reference counting: O(1) increment/decrement

---

## 🎯 Interview Preparation Roadmap

### 📅 4-Week Study Plan

<details>
<summary><b>Week 1: Foundations (Beginner Systems)</b> - Click to expand</summary>

**Goal**: Build confidence with core OOP and data structures

| Day | Morning (2h) | Evening (2h) | Deliverable |
|-----|--------------|--------------|-------------|
| Mon | Study Cache System README | Implement LRU from scratch | Working LRU cache |
| Tue | Study Parking Lot README | Implement basic parking system | Spot allocation logic |
| Wed | Study Library System README | Implement book checkout | Fine calculation working |
| Thu | Study Snake & Ladder README | Implement game logic | Complete game playable |
| Fri | Review all 4 systems | Mock interview (self-record) | Video recording |
| Sat | Refactor code, add tests | Write design doc for 1 system | Documentation |
| Sun | Rest or light review | Prepare questions for week 2 | Question list |

**Week 1 Checkpoint**: Can you implement LRU cache in 20 minutes?

</details>

<details>
<summary><b>Week 2: Algorithms (Intermediate Systems)</b> - Click to expand</summary>

**Goal**: Master algorithmic thinking and state machines

| Day | Morning (2h) | Evening (2h) | Deliverable |
|-----|--------------|--------------|-------------|
| Mon | Study Elevator System | Implement SCAN algorithm | Working scheduler |
| Tue | Study Vending Machine | Implement state machine | Complete FSM |
| Wed | Study URL Shortener | Implement encoding/decoding | Working shortener |
| Thu | Study Chess Game | Implement move validation | Valid moves only |
| Fri | Practice all 4 systems | Timed coding (45 min each) | 4 implementations |
| Sat | Study design patterns used | Refactor with patterns | Cleaner code |
| Sun | Mock interview with peer | Get feedback | Feedback notes |

**Week 2 Checkpoint**: Can you explain SCAN algorithm and when to use State pattern?

</details>

<details>
<summary><b>Week 3: Distributed Systems (Advanced)</b> - Click to expand</summary>

**Goal**: Understand distributed system concepts

| Day | Morning (2h) | Evening (2h) | Deliverable |
|-----|--------------|--------------|-------------|
| Mon | Rate Limiter theory | Implement token bucket | Working rate limiter |
| Tue | Job Processor theory | Implement worker pool | Concurrent processing |
| Wed | Distributed Cache theory | Implement consistent hashing | Hash ring working |
| Thu | KV Store theory | Implement vector clocks | Conflict detection |
| Fri | Autocomplete theory | Implement Trie | Prefix search working |
| Sat | File Storage theory | Implement chunking | Deduplication working |
| Sun | Review CAP theorem | Compare all distributed systems | Comparison doc |

**Week 3 Checkpoint**: Can you explain CAP theorem with examples from implemented systems?

</details>

<details>
<summary><b>Week 4: Integration & Practice</b> - Click to expand</summary>

**Goal**: Polish interview skills and connect LLD to HLD

| Day | Morning (2h) | Evening (2h) | Deliverable |
|-----|--------------|--------------|-------------|
| Mon | Full mock interview | Review recording | Improvement list |
| Tue | Practice weakest system | Implement from scratch | Clean implementation |
| Wed | Study HLD connections | Map LLD to system design | HLD diagrams |
| Thu | Behavioral prep | STAR stories for each system | Story bank |
| Fri | Final mock interview | Get feedback | Ready for real interviews! |
| Sat | Review common questions | Practice explanations | Confident answers |
| Sun | Rest and confidence building | Light review only | Mental preparation |

**Week 4 Checkpoint**: Can you design any system in 45 minutes with clean code and explain trade-offs?

</details>

**Total Time Investment**: 
- **Minimum**: 60 hours (1 hour/day for 2 months)
- **Recommended**: 80-100 hours (2-3 hours/day for 1 month)
- **Intensive**: 120+ hours (4-5 hours/day for 3-4 weeks)

---

### 🎓 Phase 1: Study Individual Systems
**Goal**: Understand each system deeply

**Beginner-Friendly Order** (start here):
1. **Cache System** - Fundamental data structures (HashMap + LinkedList)
2. **Parking Lot** - Basic OOP concepts, entity relationships
3. **Library Management** - Business logic, date handling
4. **Snake and Ladder Game** - Strategy pattern, turn management, statistics

**Intermediate Systems** (algorithmic focus):
5. **Elevator System** - SCAN algorithm, state machines
6. **Vending Machine** - State pattern, payment processing
7. **URL Shortener** - Encoding, hashing, scalability
8. **Chess Game** - OOP design, move validation, complex rules

**Advanced Systems** (distributed concepts):
9. **Rate Limiter** - Distributed coordination, algorithms
10. **Job Processor** - Concurrency, reliability patterns
11. **Notification Service** - Multi-channel delivery, deduplication, retry mechanisms
12. **File System** - Tree structures, path resolution, permissions, hierarchical organization
13. **Distributed Cache** - Consistent hashing, quorum consensus, read repair
14. **Distributed KV Store** - Vector clocks, Merkle trees, WAL, MVCC
15. **Autocomplete System** - Trie data structures, caching, fuzzy matching
16. **File Storage System** - Chunking, deduplication, delta sync, versioning

**Study approach for each**:
1. Read the detailed README thoroughly
2. Understand core algorithms and data structures
3. Run the Python implementation
4. Trace through code execution with examples
5. Practice explaining design decisions out loud

---

### Phase 2: Implementation Practice (Week 2-3)
**Goal**: Code from scratch under time pressure

**Practice routine** (per system):
1. **Day 1**: Implement basic version (45 minutes)
   - Core functionality only
   - No error handling yet
   
2. **Day 2**: Add production features (45 minutes)
   - Error handling
   - Edge cases
   - Thread safety
   
3. **Day 3**: Mock interview (60 minutes)
   - Requirements clarification (5 min)
   - Design discussion (10 min)
   - Implementation (30 min)
   - Q&A and extensions (15 min)

**Key focus areas**:
- Write clean, readable code
- Think aloud while coding
- Handle edge cases
- Discuss time/space complexity

---

### Phase 3: Pattern Recognition (Week 3)
**Goal**: Recognize when to apply each design pattern

**Design Patterns across systems**:

| Pattern | Used In | Why |
|---------|---------|-----|
| **Strategy** | Cache, Rate Limiter, Elevator, Vending Machine | Multiple algorithms, runtime selection |
| **State** | Vending Machine, Elevator | Complex state transitions |
| **Factory** | Cache, Rate Limiter, Job Processor | Object creation abstraction |
| **Observer** | Library (notifications), Elevator (events) | Event-driven architecture |
| **Singleton** | Parking Lot (controller), Library (catalog) | Single instance coordination |

**Practice exercise**: Given a new problem, identify which patterns apply

---

### Phase 4: System Design Integration (Week 4)
**Goal**: Connect LLD to High-Level Design (HLD)

**Scaling exercises**:
1. **Cache System** → How does it fit into a CDN architecture?
2. **Rate Limiter** → Distributed rate limiting with Redis
3. **Job Processor** → Compare with Kafka, RabbitMQ
4. **Elevator** → IoT integration, real-time updates
5. **Vending Machine** → Fleet management, cloud backend
6. **Library** → Multi-branch coordination
7. **Parking** → City-wide parking availability system
8. **URL Shortener** → Handle 1 billion URLs, global distribution

**Consider for each**:
- Database choices (SQL vs NoSQL)
- Caching layers
- Load balancing
- Network partitions
- Monitoring and alerts

## 📊 Complexity Comparison

| System | Component | Time | Space | Key Data Structure |
|--------|-----------|------|-------|-------------------|
| **Cache** | LRU get/put | O(1) | O(n) | HashMap + Doubly Linked List |
| | LFU get/put | O(1) | O(n) | HashMap + Frequency map |
| | FIFO get/put | O(1) | O(n) | HashMap + Queue |
| **Rate Limiter** | Token Bucket | O(1) | O(users) | HashMap for user state |
| | Sliding Window | O(log n) | O(users × reqs) | Sorted set for timestamps |
| | Fixed Window | O(1) | O(users) | HashMap with counter |
| **Job Processor** | Submit job | O(log n) | O(jobs) | Priority Queue (heap) |
| | Process job | O(1) | O(workers) | Thread pool |
| | Retry job | O(1) | O(jobs) | Exponential backoff |
| **Elevator** | Add request | O(1) | O(floors) | Set for up/down requests |
| | Move (SCAN) | O(1) | O(floors) | Set operations |
| | Dispatch | O(m) | O(1) | Iterate m elevators |
| **Vending Machine** | Select product | O(1) | O(slots) | HashMap for inventory |
| | Calculate change | O(d) | O(d) | Greedy (d = denominations) |
| | Process payment | O(1) | O(1) | Payment processor call |
| **Library** | Search by title | O(log n) | O(n) | TreeMap / B-tree index |
| | Checkout book | O(1) | O(books) | HashMap lookup |
| | Calculate fine | O(1) | O(1) | Date arithmetic |
| **Parking Lot** | Find spot | O(n) worst | O(spots) | Linear search (can optimize) |
| | Calculate price | O(1) | O(1) | Time difference calc |
| | Check availability | O(1) | O(floors) | Counter per floor/type |
| **URL Shortener** | Shorten URL | O(1) avg | O(urls) | HashMap for URL→code |
| | Resolve URL | O(1) | O(urls) | HashMap for code→URL |
| | Base62 encode | O(log n) | O(1) | Convert number to base62 |
| **Chess Game** | Move validation | O(n²) | O(1) | Check detection scan |
| | Get legal moves | O(n³) | O(n²) | Iterate pieces × moves × validate |
| | Make move | O(n²) | O(n) | Validate + update board |
| **Notification** | Send notification | O(1) | O(users) | Priority queue insertion |
| | Check duplicate | O(1) | O(window) | Hash-based deduplication |
| | Check rate limit | O(log n) | O(users × reqs) | Sliding window cleanup |
| | Retry calculation | O(1) | O(1) | Exponential backoff formula |
| **File System** | Path resolution | O(d) | O(paths) | Tree traversal (d = depth) |
| | Path normalization | O(n) | O(d) | Stack for . and .. (n = path length) |
| | Create/delete file | O(d) | O(1) | Navigate to parent |
| | Tree traversal | O(n) | O(h) or O(w) | DFS (h = height) or BFS (w = width) |
| | Search by name | O(n) | O(h) | DFS with filtering |
| | Move (cycle check) | O(d) | O(1) | Traverse up to check ancestor |

## 🎤 Common Interview Questions Across All Systems

### Design & Architecture
**Q**: "Walk me through your high-level design approach"
**A**: Start with requirements clarification → Core components → Data flow → Scale considerations

**Q**: "How would you handle this at scale (millions of users)?"
**A**: Discuss sharding, distributed storage, load balancing, caching layers

**Q**: "What are the failure modes and how do you handle them?"
**A**: Network partitions, server crashes, data corruption → Monitoring, alerts, graceful degradation

### Technical Deep Dive
**Q**: "Why did you choose this data structure over alternatives?"
**A**: Analyze time/space complexity, explain trade-offs, mention alternatives

**Q**: "How do you ensure thread safety?"
**A**: Locks, lock-free structures, immutable data, actor model

**Q**: "How do you test these systems?"
**A**: Unit tests, integration tests, load tests, chaos engineering

### Production Concerns
**Q**: "How do you monitor these systems in production?"
**A**: Metrics (throughput, latency, errors), logs, dashboards, alerts

**Q**: "How do you deploy changes safely?"
**A**: Blue-green deployment, canary releases, feature flags, rollback plans

**Q**: "How do you debug performance issues?"
**A**: Profiling, distributed tracing, metrics analysis, load testing

## ⚡ Quick Reference - Key Points to Remember

### Cache System
- **LRU**: HashMap + Doubly Linked List for O(1) operations
- **Thread Safety**: Use RLock for reentrant operations
- **TTL**: Store expiration timestamp, check on access
- **Memory Management**: Implement cleanup for expired entries

### Rate Limiter  
- **Token Bucket**: Best balance of accuracy and burst handling
- **Distributed**: Use Redis with atomic operations (INCR + EXPIRE)
- **Accuracy vs Performance**: Sliding window (accurate) vs Fixed window (fast)
- **Failure Handling**: Fail open (availability) vs fail closed (protection)

### Job Processor
- **Concurrency**: ThreadPoolExecutor for worker management
- **Reliability**: Persistent job storage, retry with exponential backoff
- **Resource Management**: Job timeouts, memory limits, circuit breakers
- **Monitoring**: Track queue depth, processing rates, error rates

### Snake and Ladder Game
- **Strategy Pattern**: Inject different dice strategies for testing and game modes
- **Sparse Data**: Use HashMap for O(1) snake/ladder lookups
- **Boundary Logic**: Exact roll required to win (overshooting keeps you at current position)
- **Thread Safety**: RLock for concurrent multiplayer access

### Distributed Cache
- **Consistent Hashing**: Minimal remapping on node changes (K/N keys vs all keys with modulo)
- **Quorum Math**: W + R > N ensures read-after-write consistency
- **Read Repair**: Fix stale replicas during reads (no extra RPCs)
- **Hot Keys**: Detect via access frequency, mitigate with local caching or key splitting
- **Virtual Nodes**: Better load distribution (150 virtual per physical node)

### Distributed KV Store
- **Vector Clocks**: Track causality, detect concurrent updates (happens-before vs concurrent)
- **Merkle Trees**: O(log N) anti-entropy sync vs O(N) full comparison
- **Write-Ahead Log**: Durability guarantee, crash recovery with replay
- **Quorum Consensus**: R + W > N ensures consistency (tunable CAP trade-offs)
- **Conflict Resolution**: Reconcile concurrent writes with version vectors

### Autocomplete System
- **Trie Data Structure**: O(M) insert/search where M = word length
- **Top-K with Heap**: O(N log K) for K suggestions from N words
- **LRU Cache**: Cache popular prefixes for <1ms latency
- **Fuzzy Matching**: Levenshtein distance with O(M*N) DP for typo correction
- **Frequency Ranking**: Update on selection for personalization

### File Storage System
- **Content-Addressable**: SHA-256 chunk IDs for automatic deduplication
- **Delta Sync**: Transfer only changed chunks (rsync algorithm), 90% bandwidth savings
- **Chunking**: Fixed or variable-size (Rabin fingerprinting for better dedup)
- **Version Control**: Immutable snapshots with parent pointers
- **Conflict Resolution**: Last-write-wins or 3-way merge for concurrent edits

## 🔧 Running the Code

### Prerequisites
```bash
python 3.8+  # All systems use modern Python features
# No external dependencies required - uses only standard library
```

### Execution
```bash
# Run individual systems
python 01_cache_system.py
python 02_rate_limiter.py
python 03_job_processor.py
python 04_elevator_system.py
python 05_vending_machine.py
python 06_library_management.py
python 07_parking_lot.py
python 08_url_shortener.py
python 09_chess_game.py
python 10_notification_service.py
python 11_file_system.py
python 12_snake_ladder_game.py
python 13_distributed_cache.py
python 14_distributed_kv_store.py
python 15_autocomplete_system.py
python 16_file_storage_system.py

# Each will run comprehensive demos showing all features
```

### Expected Output
Each system includes:
- **Demonstration scenarios** showing core functionality
- **Edge case handling** (errors, boundary conditions)
- **Performance metrics** (operations per second, memory usage)
- **Example transactions** with detailed logs

### Customization Examples

**Cache System**:
```python
from cache_system import CacheFactory, EvictionPolicy

cache = CacheFactory.create(
    EvictionPolicy.LRU, 
    capacity=1000, 
    default_ttl=300  # 5 minutes
)
cache.put("key1", "value1")
value = cache.get("key1")
```

**Rate Limiter**:
```python
from rate_limiter import RateLimiterFactory, RateLimitStrategy

limiter = RateLimiterFactory.create(
    RateLimitStrategy.TOKEN_BUCKET,
    max_requests=100,
    time_window_seconds=60
)
allowed = limiter.allow_request("user123")
```

**Job Processor**:
```python
from job_processor import JobProcessorFactory, ProcessingStrategy

processor = JobProcessorFactory.create(
    ProcessingStrategy.PRIORITY,
    max_workers=8,
    max_queue_size=10000
)
job_id = processor.submit_job(my_function, priority=1)
```

**Elevator System**:
```python
from elevator_system import ElevatorController, DispatchStrategy

controller = ElevatorController(
    num_elevators=4,
    num_floors=20,
    strategy=DispatchStrategy.OPTIMIZED
)
controller.request_elevator(floor=5, direction="UP")
```

**Vending Machine**:
```python
from vending_machine import VendingMachine, PaymentType

machine = VendingMachine("VM001")
machine.add_inventory_slot(slot_id="A1", product=my_product, quantity=10)
machine.insert_payment(5.00, PaymentType.CASH)
success, msg = machine.select_product("A1")
```

**Library Management**:
```python
from library_management import Library, Book, Member

library = Library("City Library")
book = Book(isbn="978-0-123456-78-9", title="Design Patterns", author="Gang of Four")
library.add_book(book)

member = Member(member_id="M001", name="John Doe", member_type="PREMIUM")
library.checkout_book(member.member_id, book.isbn)
```

**Parking Lot**:
```python
from parking_lot import ParkingLot, VehicleType, PricingStrategy

parking_lot = ParkingLot(
    name="Downtown Parking",
    num_floors=5,
    spots_per_floor=50,
    pricing_strategy=PricingStrategy.HOURLY
)
ticket = parking_lot.park_vehicle(vehicle, VehicleType.COMPACT)
fee = parking_lot.unpark_vehicle(ticket.ticket_id)
```

**URL Shortener**:
```python
from url_shortener import URLShortener, EncodingStrategy

shortener = URLShortener(
    base_url="https://short.url/",
    strategy=EncodingStrategy.BASE62
)
short_url = shortener.shorten("https://example.com/very/long/url")
original_url = shortener.resolve(short_url)
```

**Chess Game**:
```python
from chess_game import ChessGame

game = ChessGame("game_001")
# Make moves using algebraic notation
success, message = game.make_move("e2", "e4")
print(message)  # "Move: e4"

# Get valid moves for a piece
valid_moves = game.get_valid_moves("e2")
print(valid_moves)  # ['e3', 'e4']

# Check game status
status = game.get_game_status()
print(status)  # {'game_state': 'active', 'current_turn': 'white', ...}
```

**Notification Service**:
```python
from notification_service import NotificationService, NotificationChannel, NotificationPriority

service = NotificationService()
service.start_worker()

# Send email notification
notification = Notification(
    notification_id="notif_001",
    user_id="user_123",
    channel=NotificationChannel.EMAIL,
    title="Welcome!",
    message="Thanks for signing up",
    priority=NotificationPriority.HIGH
)

success, message = service.send_notification(notification)
print(message)  # "Notification queued successfully"

# Check delivery status
status = service.get_delivery_status("notif_001")
print(status)  # NotificationResult(status=SENT, ...)
```

**File System**:
```python
from file_system import FileSystem, Permission

fs = FileSystem()
user = "alice"

# Create directory structure
fs.create_directory("/home", "root")
fs.create_directory("/home/alice", user)

# Create and write to file
fs.create_file("/home/alice/readme.txt", user, "Welcome!")
success, content = fs.read_file("/home/alice/readme.txt", user)
print(content)  # "Welcome!"

# List directory
success, children = fs.list_directory("/home/alice", user)
print(children)  # ['readme.txt']

# Traverse directory tree (DFS)
paths = fs.traverse_dfs("/home", user)
for path in paths:
    print(path)  # /home, /home/alice, /home/alice/readme.txt

# Search for files
results = fs.search_by_name("readme", "/", user)
print(results)  # ['/home/alice/readme.txt']
```

**Snake and Ladder Game**:
```python
from snake_ladder_game import SnakeAndLadderGame, ControlledDice, WeightedDice

# Create game with classic board
game = SnakeAndLadderGame(board_size=100)
game.setup_classic_board()  # 10 snakes, 10 ladders

# Add players
game.add_player("p1", "Alice")
game.add_player("p2", "Bob")
game.start_game()

# Play game
while not game.get_winner():
    move = game.roll_dice_and_move()
    print(f"{move.player_id} rolled {move.dice_roll}: {move.from_position}→{move.to_position}")

# Get final standings
standings = game.get_player_standings()
for player in standings:
    print(f"{player.name}: Position {player.position}, Moves: {player.moves_count}")

# Testing with controlled dice
test_game = SnakeAndLadderGame(board_size=20)
test_game.set_dice_strategy(ControlledDice([6, 6, 8]))  # Predetermined rolls
test_game.add_player("test", "Tester")
test_game.start_game()
test_game.roll_dice_and_move()  # Always rolls 6
```

**Distributed Cache**:
```python
from distributed_cache import DistributedCache, CacheNode, ConsistencyLevel

# Create cache cluster
cache = DistributedCache(
    cluster_name="prod-cache",
    replication_factor=3,
    default_consistency=ConsistencyLevel.QUORUM
)

# Add nodes
for i in range(1, 6):
    node = CacheNode(f"node{i}", f"10.0.0.{i}", 8000 + i)
    cache.add_node(node)

# Write data with replication
result = cache.put("user:1001", {"name": "Alice", "age": 30}, ttl_seconds=300)
print(f"Write: {result.message}, Replicas: {result.replicas_written}/3")

# Read data with quorum
result = cache.get("user:1001", consistency=ConsistencyLevel.QUORUM)
print(f"Read: {result.value}, Version: {result.version}")

# Get cluster statistics
stats = cache.get_cluster_stats()
print(f"Hit rate: {stats['node_stats']['node1']['hit_rate']:.2%}")
print(f"Total reads: {stats['global_stats']['total_reads']}")
print(f"Read repairs: {stats['global_stats']['read_repairs']}")
```

**Distributed KV Store**:
```python
from distributed_kv_store import DistributedKVStore

# Create cluster
store = DistributedKVStore(cluster_name="kv-cluster", replication_factor=3)
for i in range(5):
    store.add_node(f"node{i}", f"10.0.0.{i}", 9000 + i)

# Write with quorum (W=2)
store.put("user:1001", {"name": "Alice"}, write_quorum=2)

# Read with quorum (R=2, R+W>N ensures consistency)
result = store.get("user:1001", read_quorum=2)
print(f"Value: {result.value}, Vector clock: {result.vector_clock}")

# Anti-entropy synchronization
divergent = store.anti_entropy_sync("node0", "node1")
print(f"Synchronized {len(divergent)} divergent keys")
```

**Autocomplete System**:
```python
from autocomplete_system import AutocompleteSystem

# Create system with LRU cache
system = AutocompleteSystem(cache_capacity=1000)

# Build dictionary with frequencies
system.add_words([
    ("python", 10000),
    ("pytorch", 8000),
    ("pandas", 6000),
    ("javascript", 5000)
])

# Get suggestions (ranked by frequency)
suggestions = system.get_suggestions("py", max_results=5)
# Returns: [("python", 10000), ("pytorch", 8000), ...]

# Fuzzy matching for typos
fuzzy = system.get_fuzzy_suggestions("pytohn", max_distance=2)
# Returns: [("python", 10000)] with edit distance 2

# Update frequency when user selects
system.update_frequency("pytorch", increment=100)
```

**File Storage System**:
```python
from file_storage_system import FileStorageSystem

# Create storage
storage = FileStorageSystem()

# Upload file (automatically chunked and deduplicated)
version1 = storage.upload_file("/docs/report.pdf", file_data, owner="user1")

# Update file (delta sync - only changed chunks transferred)
version2 = storage.upload_file("/docs/report.pdf", updated_data, owner="user1", parent_version=version1)

# Download specific version (time-travel)
old_data = storage.download_file(file_id, version_id=version1)

# Sync with conflict resolution
version, had_conflict = storage.sync_file("/docs/report.pdf", local_data, "user1", local_version)

# Get deduplication stats
stats = storage.get_storage_stats()
print(f"Space saved: {stats['chunk_stats']['space_savings']:.1%}")
```

## 🎭 Ace Your LLD Interview - Expert Tips

### 🎯 The Perfect 45-Minute Interview Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Clarification (5-7 minutes)                        │
│ ✓ Functional requirements  ✓ Scale  ✓ Constraints           │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: High-Level Design (8-10 minutes)                   │
│ ✓ Core components  ✓ APIs  ✓ Data flow  ✓ Diagram           │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: Deep Dive (20-25 minutes)                          │
│ ✓ Implement 2-3 key classes  ✓ Core algorithms              │
├─────────────────────────────────────────────────────────────┤
│ Phase 4: Discussion (5-10 minutes)                          │
│ ✓ Trade-offs  ✓ Extensions  ✓ Production concerns           │
└─────────────────────────────────────────────────────────────┘
```

### 💡 Winning Communication Strategies

<details>
<summary><b>✅ DO: Ask These Clarifying Questions</b></summary>

**Scale & Performance:**
- "How many users/requests per second?"
- "What's the acceptable latency?"
- "Read-heavy or write-heavy workload?"

**Functional Requirements:**
- "Should this handle concurrent access?"
- "Do we need persistence or in-memory only?"
- "What happens if system crashes?"

**Constraints:**
- "Any specific technologies/languages required?"
- "Should we consider distributed deployment?"
- "What's the priority: consistency or availability?"

</details>

<details>
<summary><b>✅ DO: Think Out Loud (Example)</b></summary>

**Bad** (Silent coding):
```
[Types code silently for 5 minutes]
"Done. Here's the cache."
```

**Good** (Narrated approach):
```
"I'll use a HashMap for O(1) lookups, and a doubly-linked 
list to track access order for LRU eviction. Let me start 
with the Node class... Now the cache class with get and put 
methods... For thread safety, I'm using a ReentrantLock 
around critical sections..."
```

</details>

<details>
<summary><b>✅ DO: Discuss Trade-offs</b></summary>

**Example: Rate Limiter Algorithm Choice**

| Algorithm | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **Token Bucket** | • Handles bursts<br>• Simple | • Less accurate | APIs with variable traffic |
| **Sliding Window** | • Most accurate | • Memory intensive | Critical rate limiting |
| **Fixed Window** | • Very fast<br>• Low memory | • Boundary issues | Non-critical scenarios |

**What to say**: *"I'd choose Token Bucket because it handles burst traffic well and is industry-standard. Sliding Window would be more accurate but uses more memory. For this scale [X requests/sec], Token Bucket is the sweet spot."*

</details>

### ❌ Fatal Mistakes That Fail Interviews

<table>
<tr>
<th width="50%">❌ Don't Do This</th>
<th width="50%">✅ Do This Instead</th>
</tr>

<tr>
<td>
<b>Jump to coding immediately</b><br>
<code>interviewer: "Design a cache"</code><br>
<code>you: [starts coding]</code>
</td>
<td>
<b>Clarify first, then design</b><br>
<code>you: "Before I start, let me understand:</code><br>
<code>- What's the eviction policy?</code><br>
<code>- Do we need thread safety?</code><br>
<code>- Memory constraints?"</code>
</td>
</tr>

<tr>
<td>
<b>Ignore edge cases</b><br>
<code>def get(key):</code><br>
<code>    return self.cache[key]</code><br>
<i>(What if key doesn't exist?)</i>
</td>
<td>
<b>Handle errors gracefully</b><br>
<code>def get(key):</code><br>
<code>    if key not in self.cache:</code><br>
<code>        return None</code><br>
<code>    return self.cache[key]</code>
</td>
</tr>

<tr>
<td>
<b>Optimize prematurely</b><br>
<i>"I'll use a B-tree with red-black balancing and..."</i><br>
(For a problem needing just a HashMap)
</td>
<td>
<b>Start simple, justify later</b><br>
<i>"I'll start with a HashMap for O(1) access. If we need ordering, we can switch to TreeMap later. For now, simplicity is key."</i>
</td>
</tr>

<tr>
<td>
<b>Ignore threading</b><br>
<code># No locks, no synchronization</code><br>
<code># "It'll be fine..."</code>
</td>
<td>
<b>Address concurrency</b><br>
<code>with self._lock:</code><br>
<code>    # Critical section</code><br>
<i>"I'm using a lock here because multiple threads might access this simultaneously."</i>
</td>
</tr>

<tr>
<td>
<b>Give up when stuck</b><br>
<i>"I don't know how to handle this..."</i><br>
[awkward silence]
</td>
<td>
<b>Work through it collaboratively</b><br>
<i>"I'm thinking through a few approaches:<br>
1. Use a queue [trade-off A]<br>
2. Use a heap [trade-off B]<br>
What do you think?"</i>
</td>
</tr>

</table>

### 🏆 Pro Tips from Successful Candidates

1. **Write testable code**: After implementing, say *"Let me trace through an example: user requests X, we check Y, return Z"*

2. **Use design patterns naturally**: Don't force them, but when appropriate say *"This is a perfect use case for Strategy pattern because..."*

3. **Show production thinking**: *"In production, I'd add logging here, metrics there, and circuit breaker for this external call"*

4. **Handle time pressure**: If running out of time, say *"I'll implement the core logic first, then we can discuss error handling and edge cases"*

5. **Ask for hints**: If genuinely stuck, ask *"I'm debating between approach A and B. Any preference or hints on which direction to explore?"*

### 📝 Post-Interview Action Items

After each mock or real interview:
- [ ] Record yourself and watch (cringe-worthy but effective!)
- [ ] List 3 things you did well
- [ ] List 3 areas to improve
- [ ] Research any concepts you struggled with
- [ ] Re-implement the system cleanly within 30 minutes

---

## 🚀 Beyond the Basics - Advanced Topics

### Advanced Topics to Explore
1. **Consistency Models**: Strong vs Eventual consistency, CAP theorem
2. **Distributed Consensus**: Raft, Paxos algorithms
3. **Data Replication**: Master-slave, master-master, multi-master
4. **Partitioning Strategies**: Hash-based, range-based, directory-based
5. **Circuit Breakers**: Fault tolerance and graceful degradation
6. **Event Sourcing**: Audit trails and state reconstruction
7. **CQRS**: Command Query Responsibility Segregation
8. **Saga Pattern**: Distributed transaction management

### Additional LLD Problems to Practice
**Beginner Level**:
1. **ATM System** - State machines, transaction handling
2. **Snake Game** - Game loops, collision detection
3. **Tic-Tac-Toe** - Board representation, win detection
4. **Stack Overflow Clone** - Voting, reputation, tags

**Intermediate Level**:
5. **Online Shopping Cart** - Inventory, checkout, pricing
6. **Movie Ticket Booking** - Seat allocation, concurrent bookings
7. **Ride-Sharing App** - Matching, pricing, routing
8. **Social Media Feed** - Timeline generation, ranking algorithms

**Advanced Level**:
9. **Hotel Management System** - Reservations, room allocation
10. **Stock Exchange** - Order matching, price discovery
11. **Splitwise / Expense Sharing** - Debt simplification algorithms
12. **Google Calendar** - Event scheduling, conflicts, recurring events

### System Design Integration
**Connect LLD to HLD**:
- **Cache System** → CDN architecture, multi-level caching
- **Rate Limiter** → API gateway, DDoS protection
- **Job Processor** → Message queues (Kafka, RabbitMQ)
- **Elevator** → IoT systems, real-time coordination
- **Vending Machine** → Fleet management, cloud integration
- **Library** → Multi-branch, federation, inter-library loans
- **Parking Lot** → City-wide availability, mobile apps
- **URL Shortener** → Global distribution, DNS, load balancing
- **Chess Game** → Online multiplayer, AI opponents, game replay

### Study Resources

**Books**:
- *"Designing Data-Intensive Applications"* - Martin Kleppmann ⭐⭐⭐⭐⭐
- *"Head First Design Patterns"* - Freeman & Freeman
- *"System Design Interview"* Vol 1 & 2 - Alex Xu
- *"Clean Code"* - Robert C. Martin
- *"Refactoring"* - Martin Fowler

### 📚 Recommended Resources

<details>
<summary><b>📖 Books (Must-Reads)</b></summary>

**Low Level Design:**
1. **"Head First Design Patterns"** - Freeman & Freeman  
   *Best for: Understanding patterns through real examples*  
   ⭐⭐⭐⭐⭐ - Start here if new to design patterns

2. **"Clean Code"** - Robert C. Martin  
   *Best for: Writing maintainable, readable code*  
   ⭐⭐⭐⭐⭐ - Essential for interview coding

3. **"Refactoring"** - Martin Fowler  
   *Best for: Improving existing code, recognizing code smells*  
   ⭐⭐⭐⭐ - Great for mid-level+ engineers

**System Design (HLD Context):**
4. **"Designing Data-Intensive Applications"** - Martin Kleppmann  
   *Best for: Understanding distributed systems deeply*  
   ⭐⭐⭐⭐⭐ - The bible for system design

5. **"System Design Interview Vol 1 & 2"** - Alex Xu  
   *Best for: Interview preparation with practical examples*  
   ⭐⭐⭐⭐⭐ - Most relevant for interviews

</details>

<details>
<summary><b>🎓 Online Courses</b></summary>

**Paid Courses:**
- [**Grokking the Object Oriented Design Interview**](https://www.educative.io) (Educative)  
  *15-20 hours | $79 | Best structured LLD course*

- [**Master the Coding Interview: Data Structures + Algorithms**](https://www.udemy.com) (Udemy)  
  *22 hours | $15-20 | Good DS/Algo foundation*

**Free Resources:**
- [**System Design Primer**](https://github.com/donnemartin/system-design-primer) (GitHub)  
  *Comprehensive guide with examples*

- [**Coding Interview University**](https://github.com/jwasham/coding-interview-university) (GitHub)  
  *Complete CS degree roadmap*

</details>

<details>
<summary><b>🎥 YouTube Channels</b></summary>

1. **Gaurav Sen** - System design explanations  
2. **Tech Dummies Narendra L** - LLD focused  
3. **Exponent** - Mock interviews and frameworks  
4. **interviewing.io** - Real interview recordings  
5. **Clément Mihailescu** - AlgoExpert founder, great explanations  

</details>

<details>
<summary><b>💻 Practice Platforms</b></summary>

| Platform | Focus | Best For | Cost |
|----------|-------|----------|------|
| [LeetCode](https://leetcode.com) | Algorithms | FAANG prep | Free/Premium |
| [AlgoExpert](https://algoexpert.io) | Curated problems | Structured learning | $99/year |
| [Educative](https://educative.io) | Interactive courses | Design patterns | $18-79/course |
| [Pramp](https://pramp.com) | Mock interviews | Practice with peers | Free |
| [interviewing.io](https://interviewing.io) | Anonymous mocks | Real interview practice | Free |

</details>

<details>
<summary><b>🏢 Company-Specific Prep</b></summary>

**FAANG Focus:**
- Amazon: Study **Leadership Principles**, focus on Rate Limiter, Job Processor  
- Google: Focus on **scalability**, study Distributed Cache, KV Store  
- Meta: Focus on **social features**, study Notification Service, File Storage  
- Apple: Focus on **design patterns**, study Cache, Vending Machine, State machines  
- Netflix: Focus on **distributed systems**, study all distributed systems (#13, #14, #2)

**Leetcode Problem Patterns:**
- Top interview questions (75 most common)
- Design-specific tags (Design, OOD)
- Company-tagged problems

</details>

---

## 🤝 Contributing & Community

### How to Contribute

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs**: Open an issue with details
2. **💡 Suggest Improvements**: Share your ideas for better explanations
3. **📝 Add Documentation**: Improve READMEs, add examples
4. **🆕 New Systems**: Propose additional LLD problems
5. **✅ Add Tests**: Unit tests, integration tests

**Contribution Guidelines:**
- Follow existing code structure
- Include comprehensive demos
- Add interview Q&A sections
- Document complexity analysis
- Ensure thread safety where needed

### Community & Support

- **💬 Discussions**: Use GitHub Discussions for questions
- **⭐ Star**: If this helped you, please star the repo!
- **🔀 Fork**: Create your own variations
- **📢 Share**: Help others find this resource

### 📊 Repository Stats

- **16 Complete Systems** with production-ready code
- **6,000+ lines** of documented Python code
- **80+ hours** of structured learning content
- **100+ interview Q&As** with detailed answers
- **50+ design patterns** demonstrated across systems

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**You are free to:**
- ✅ Use for interview preparation
- ✅ Fork and modify
- ✅ Use in educational settings
- ✅ Share with colleagues

**Please:**
- ⭐ Star the repo if it helped you
- 📝 Credit the source when sharing
- 🤝 Contribute improvements back

---

## 🙏 Acknowledgments

**Inspired by:**
- Real interview experiences from FAANG companies
- Common patterns from 100+ LLD interviews
- Community feedback and suggestions
- Industry best practices and production systems

**Special Thanks:**
- To all contributors who improve this resource
- To the companies that share their interview processes
- To the open-source community for amazing tools

---

## 🎓 Final Words

> **"The key to acing LLD interviews isn't memorizing solutions—it's understanding principles and being able to apply them to new problems."**

**Your Journey:**
1. ✅ **Week 1-2**: Study beginner systems, build confidence
2. ✅ **Week 3-4**: Master intermediate + advanced systems
3. ✅ **Week 5**: Polish through mock interviews
4. ✅ **Week 6**: You're ready! Apply to companies

**Remember:**
- 💪 Consistent practice beats cramming
- 🗣️ Communication skills matter as much as coding
- 🧠 Understanding trade-offs shows senior thinking
- 🤝 Collaborate with interviewers, don't work in silence
- 😌 Stay calm—you've prepared well!

---

<div align="center">

### 🚀 Ready to Ace Your Interviews?

**Start with System #1 (Cache) → Build momentum → Master all 16**

[⬆️ Back to Top](#-low-level-design-lld-interview-mastery) | [📖 View Systems](#-complete-system-catalog) | [🎯 Study Plan](#-interview-preparation-roadmap)

---

**Made with ❤️ for interview preparation**

*Last Updated: December 2025 | Version 2.0*

**⭐ If this helped you land your dream job, please star the repo! ⭐**

</div>
- Grokking the System Design Interview (Educative)
- System Design Primer (GitHub - donnemartin)

**Papers**:
- Google MapReduce
- Amazon Dynamo
- Facebook TAO
- Google Bigtable
- Apache Kafka architecture

**Blogs**:
- High Scalability (highscalability.com)
- AWS Architecture Blog
- Netflix Tech Blog
- Uber Engineering Blog
- Martin Fowler's Blog

**Practice Platforms**:
- LeetCode (System Design section)
- InterviewBit
- Pramp (peer mock interviews)
- SystemsExpert (AlgoExpert)

### Interview Preparation Checklist

**Technical Skills**:
- [ ] Can implement LRU cache from scratch in 20 minutes
- [ ] Understand all 8 systems deeply
- [ ] Can explain trade-offs for each design decision
- [ ] Know time/space complexity of all operations
- [ ] Comfortable with 5+ design patterns

**Communication Skills**:
- [ ] Can articulate design decisions clearly
- [ ] Ask clarifying questions before coding
- [ ] Think aloud while solving problems
- [ ] Handle feedback and pivot gracefully
- [ ] Explain complex concepts simply

**System Design Knowledge**:
- [ ] Understand scaling from 1 to 1 million users
- [ ] Know database choices (SQL vs NoSQL)
- [ ] Familiar with caching strategies
- [ ] Understand load balancing and sharding
- [ ] Know monitoring and alerting best practices

---

## 💡 Final Thoughts

These **11 systems** cover the fundamental building blocks of software engineering interviews:

**Core Concepts**:
- **Caching** → Performance optimization, data access patterns
- **Rate Limiting** → Resource protection, traffic management  
- **Job Processing** → Asynchronous execution, reliability
- **Elevator** → Algorithm design, state machines
- **Vending Machine** → State patterns, payment processing
- **Library** → Entity relationships, business logic
- **Parking Lot** → Resource allocation, pricing strategies
- **URL Shortener** → Encoding, hashing, scalability
- **Chess Game** → OOP design, complex rule validation, game state management
- **Notification Service** → Multi-channel delivery, deduplication, reliability patterns
- **File System** → Tree structures, path resolution, permissions, hierarchical organization

**What Makes a Strong Candidate**:
1. **Problem-solving approach** - Systematic, structured thinking
2. **Trade-off analysis** - No solution is perfect, discuss alternatives
3. **Communication** - Explain reasoning clearly
4. **Code quality** - Clean, readable, maintainable
5. **Production mindset** - Error handling, monitoring, scalability

**Remember**: The goal is to demonstrate your **engineering thinking**, not just write code. Interviewers want to see:
- How you break down problems
- How you make design decisions
- How you handle requirements changes
- How you think about edge cases
- How you communicate technical concepts

**Success Formula**:
```
Interview Success = Problem Solving (40%) 
                  + Communication (30%) 
                  + Code Quality (20%) 
                  + System Thinking (10%)
```

**Good luck with your interviews!** 🎯

---

## 📝 Repository Structure

```
.
├── README.md                          # This file
├── 01_cache_system.py                 # LRU, LFU, FIFO cache implementations
├── 01_cache_system_readme.md          # Detailed cache guide
├── 02_rate_limiter.py                 # Token bucket, sliding window, fixed window
├── 02_rate_limiter_readme.md          # Detailed rate limiter guide
├── 03_job_processor.py                # Job scheduling, worker pools, retry logic
├── 03_job_processor_readme.md         # Detailed job processor guide
├── 04_elevator_system.py              # SCAN algorithm, multi-elevator dispatch
├── 04_elevator_system_readme.md       # Detailed elevator guide
├── 05_vending_machine.py              # State pattern, payment processing
├── 05_vending_machine_readme.md       # Detailed vending machine guide
├── 06_library_management.py           # Book checkout, fines, reservations
├── 06_library_management_readme.md    # Detailed library guide
├── 07_parking_lot.py                  # Spot allocation, pricing, capacity
├── 07_parking_lot_readme.md           # Detailed parking lot guide
├── 08_url_shortener.py                # Base62 encoding, analytics, expiration
├── 08_url_shortener_readme.md         # Detailed URL shortener guide
├── 09_chess_game.py                   # Chess game with all rules and special moves
├── 09_chess_game_readme.md            # Detailed chess game guide
├── 10_notification_service.py         # Multi-channel notifications, deduplication, retry
├── 10_notification_service_readme.md  # Detailed notification service guide
├── 11_file_system.py                  # Hierarchical file system with permissions and traversal
└── 11_file_system_readme.md           # Detailed file system guide
```

---

## 🤝 Contributing

Found an issue or want to add a new system? Contributions are welcome!

**How to contribute**:
1. Fork the repository
2. Create a feature branch
3. Add your implementation + detailed README
4. Ensure code follows existing patterns
5. Submit a pull request

**What we're looking for**:
- Clean, well-documented code
- Comprehensive interview guides
- Real-world scenarios and edge cases
- Production considerations

---

## 📄 License

This repository is for educational purposes. Feel free to use for interview preparation and learning.

---

**Made with ❤️ for aspiring software engineers**

*Last Updated: November 2025*