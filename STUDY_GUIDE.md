# 📖 Complete LLD Interview Study Guide

<div align="center">

**Your Personalized Path to Low Level Design Mastery**

*A comprehensive, step-by-step guide to ace your LLD interviews*

</div>

---

## 🎯 How to Use This Guide

This study guide is designed for **self-paced learning** with realistic time commitments. Whether you have 2 weeks, 1 month, or 3 months, we've got you covered.

**Study Guide Structure:**
- ✅ **Daily learning objectives** with clear outcomes
- ✅ **Hands-on exercises** for each concept
- ✅ **Self-assessment quizzes** to track progress
- ✅ **Mock interview questions** with model answers
- ✅ **Common pitfalls** and how to avoid them
- ✅ **Progress tracking** with checkpoints

---

## 📅 Choose Your Timeline

### Option 1: Intensive (2 Weeks) ⚡
**Time Commitment**: 5-6 hours/day (70-80 hours total)  
**Best For**: Upcoming interviews, bootcamp grads, quick refresher  
**Outcome**: Cover all 16 systems with functional understanding

### Option 2: Balanced (4 Weeks) ⭐ RECOMMENDED
**Time Commitment**: 3-4 hours/day (80-100 hours total)  
**Best For**: Working professionals, systematic learners  
**Outcome**: Deep understanding + practice + mock interviews

### Option 3: Thorough (8-12 Weeks) 🎓
**Time Commitment**: 1-2 hours/day (60-100 hours total)  
**Best For**: Students, career switchers, building from scratch  
**Outcome**: Master-level understanding + extensive practice

---

# 📚 PART 1: FOUNDATIONS (Week 1-2)

## Week 1: Core Concepts & Beginner Systems

### Day 1: Data Structures & OOP Refresher

<details>
<summary><b>📖 Learning Objectives</b></summary>

By end of day, you should be able to:
- [ ] Explain time/space complexity for common operations
- [ ] Draw and implement LinkedList, HashMap, Tree structures
- [ ] Understand OOP principles (Encapsulation, Inheritance, Polymorphism, Abstraction)
- [ ] Know when to use which data structure

</details>

<details>
<summary><b>📝 Study Materials (2 hours)</b></summary>

**Data Structures Review:**

```python
# Essential Data Structures Cheat Sheet

# 1. HashMap / Dictionary - O(1) average case
cache = {}
cache["key"] = "value"         # O(1)
value = cache.get("key")        # O(1)
"key" in cache                  # O(1)

# 2. Doubly Linked List - For LRU Cache
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

# 3. Heap / Priority Queue - For Job Processor
import heapq
heap = []
heapq.heappush(heap, (priority, item))  # O(log n)
item = heapq.heappop(heap)               # O(log n)

# 4. TreeMap / Sorted Dict - For Library System
from sortedcontainers import SortedDict
sorted_dict = SortedDict()

# 5. Trie - For Autocomplete
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

**OOP Principles:**

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

### Day 4-5: Additional Beginner Systems

*Continue with Library Management, Snake & Ladder Game using the same detailed structure...*

---

## Week 2: Intermediate Systems & Algorithms

### Day 8: Elevator System - SCAN Algorithm Deep Dive

<details>
<summary><b>📖 Algorithm Masterclass</b></summary>

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
