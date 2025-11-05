# Low Level Design (LLD) Interview Practice - Complete Guide

## 📚 Overview
This repository contains **9 production-ready implementations** of common Low Level Design interview problems, complete with comprehensive interview guides and best practices. Each system demonstrates core OOP principles, design patterns, and real-world considerations.

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

## 🎯 Interview Preparation Strategy

### Phase 1: Study Individual Systems (Week 1-2)
**Goal**: Understand each system deeply

**Beginner-Friendly Order** (start here):
1. **Cache System** - Fundamental data structures (HashMap + LinkedList)
2. **Parking Lot** - Basic OOP concepts, entity relationships
3. **Library Management** - Business logic, date handling

**Intermediate Systems** (algorithmic focus):
4. **Elevator System** - SCAN algorithm, state machines
5. **Vending Machine** - State pattern, payment processing
6. **URL Shortener** - Encoding, hashing, scalability
7. **Chess Game** - OOP design, move validation, complex rules

**Advanced Systems** (distributed concepts):
8. **Rate Limiter** - Distributed coordination, algorithms
9. **Job Processor** - Concurrency, reliability patterns

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

## 📈 Interview Performance Tips

### Coding Best Practices
1. **Start simple, iterate**: Basic implementation → Add features
2. **Explain as you code**: Verbalize your thought process
3. **Handle edge cases**: Null inputs, boundary conditions
4. **Consider error cases**: What can go wrong?
5. **Think about testing**: How would you verify correctness?

### Communication Strategy
1. **Ask clarifying questions**: Requirements, constraints, scale
2. **Discuss trade-offs**: Every design decision has alternatives
3. **Think out loud**: Share your reasoning process
4. **Consider production**: Monitoring, deployment, maintenance
5. **Stay calm under pressure**: Break problems into smaller parts

### Common Mistakes to Avoid
❌ **Don't**: Jump straight to coding without understanding requirements  
✅ **Do**: Spend 20% of time on requirements clarification

❌ **Don't**: Ignore thread safety and concurrency  
✅ **Do**: Discuss threading model and synchronization

❌ **Don't**: Focus only on happy path  
✅ **Do**: Consider failure scenarios and edge cases

❌ **Don't**: Optimize prematurely  
✅ **Do**: Start with correct implementation, then optimize

❌ **Don't**: Ignore operational concerns  
✅ **Do**: Discuss monitoring, scaling, deployment

## 🚀 Next Steps After Mastering These Systems

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

**Online Courses**:
- Grokking the Object Oriented Design Interview (Educative)
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

These **9 systems** cover the fundamental building blocks of software engineering interviews:

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
└── 09_chess_game_readme.md            # Detailed chess game guide
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