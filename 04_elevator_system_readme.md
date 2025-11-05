# Elevator System - Interview Guide

## 📋 Overview
A multi-elevator control system implementing SCAN algorithm, optimized dispatching strategies, and peak load handling with thread-safe concurrent operations.

## 🎯 Interview Focus Areas

### Core Concepts to Master
1. **SCAN Algorithm**: Elevator scheduling (borrowed from OS disk scheduling)
2. **Dispatch Strategies**: How to select best elevator for a request
3. **Concurrency**: Thread-safe multi-elevator coordination
4. **Optimization**: Load balancing and efficiency under constraints
5. **Design Patterns**: Strategy pattern for dispatch algorithms

## 🔥 Step-by-Step Implementation Guide

### Phase 1: Requirements Clarification (2-3 minutes)
**Always ask these questions:**
```
Q: How many floors in the building?
Q: How many elevators do we need to support?
Q: What's the passenger capacity per elevator?
Q: Do we need to handle concurrent requests?
Q: Should we optimize for any specific scenario (e.g., morning rush)?
Q: Do we need to track metrics/analytics?
```

### Phase 2: High-Level Design (3-4 minutes)
1. **Draw the architecture**:
   ```
   ElevatorController (Central Coordinator)
   ├── Elevator 1 (SCAN algorithm)
   ├── Elevator 2 (SCAN algorithm)
   └── Elevator N (SCAN algorithm)
   
   DispatchStrategy (Strategy Pattern)
   ├── NearestCarStrategy
   ├── OptimizedStrategy
   └── ZoneBasedStrategy
   ```

2. **Explain key components**:
   - **Elevator**: Individual car with SCAN algorithm for efficient movement
   - **ElevatorController**: Coordinates multiple elevators, dispatches requests
   - **DispatchStrategy**: Pluggable algorithm for selecting best elevator
   - **Request Queue**: Manages pending requests during high load

### Phase 3: Implementation (15-20 minutes)

#### Start with Single Elevator + SCAN Algorithm
```python
class Elevator:
    def __init__(self, elevator_id, num_floors):
        self.current_floor = 1
        self.direction = Direction.IDLE
        # SCAN uses separate queues for up/down requests
        self.up_requests = set()    # Floors to visit going up
        self.down_requests = set()  # Floors to visit going down
```

**🎯 Interview Tip**: Explain SCAN algorithm clearly:
- "SCAN is borrowed from disk scheduling - elevator continues in one direction until no more requests, then reverses"
- "This prevents starvation and minimizes direction changes"
- "Time complexity: O(1) for movement, O(log n) for request insertion"

#### Key Methods to Implement:
1. **add_request()**: Add floor to appropriate queue (up/down)
2. **move()**: Execute one step of SCAN algorithm
3. **_move_up() / _move_down()**: Handle direction-specific movement

## 📚 Critical Knowledge Points

### 1. SCAN Algorithm Implementation
```python
def move(self):
    """
    SCAN Algorithm Steps:
    1. If IDLE, pick direction based on requests
    2. Continue in current direction
    3. Service all requests in path
    4. When no more requests in direction, reverse
    5. If no requests at all, go IDLE
    """
    if self.direction == Direction.UP:
        return self._move_up()
    else:
        return self._move_down()

def _move_up(self):
    # Stop at current floor if requested
    if self.current_floor in self.up_requests:
        self.up_requests.remove(self.current_floor)
        return True
    
    # Move to next floor if more requests above
    if self.up_requests and max(self.up_requests) > self.current_floor:
        self.current_floor += 1
        return True
    
    # No more up requests - change direction or idle
    if self.down_requests:
        self.direction = Direction.DOWN
        return self._move_down()
    else:
        self.direction = Direction.IDLE
        return False
```

### 2. Dispatch Strategy Pattern
```python
# Strategy Pattern - allows runtime algorithm switching
class DispatchStrategy(ABC):
    @abstractmethod
    def select_elevator(self, elevators, request):
        pass

class NearestCarStrategy(DispatchStrategy):
    """Simplest: choose closest elevator"""
    def select_elevator(self, elevators, request):
        return min(elevators, 
                  key=lambda e: abs(e.current_floor - request.floor))

class OptimizedStrategy(DispatchStrategy):
    """Consider distance, direction, and load"""
    def select_elevator(self, elevators, request):
        def score(elevator):
            # Lower score is better
            distance_score = abs(elevator.current_floor - request.floor)
            
            # Penalty if moving away from request
            if elevator.direction != Direction.IDLE:
                is_aligned = self._check_direction_alignment(elevator, request)
                if not is_aligned:
                    distance_score += 20  # Heavy penalty
            
            # Prefer less loaded elevators
            load_score = elevator.current_load / elevator.capacity
            
            return distance_score + load_score * 5
        
        return min(elevators, key=score)
```

### 3. Concurrency Handling
```python
class Elevator:
    def __init__(self, ...):
        self.lock = RLock()  # Reentrant lock for nested calls
    
    def add_request(self, floor):
        with self.lock:  # Ensures thread-safe operations
            if floor > self.current_floor:
                self.up_requests.add(floor)
            else:
                self.down_requests.add(floor)
```

### 4. Peak Load Optimization
```python
# Morning rush: everyone goes UP from lobby
# Solution: Batch loading at ground floor
def optimize_for_peak_load(self):
    if self._is_morning_rush():
        # Stage multiple elevators at ground floor
        # Load to capacity before departing
        # Use express elevators for high floors
        pass
```

## ⚡ Performance Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| add_request() | O(log n) | O(n) |
| move() | O(1) | O(1) |
| select_elevator() | O(m) where m = num_elevators | O(1) |
| Overall System | O(m + log n) per request | O(n * m) |

**Key Insights:**
- SCAN algorithm is O(1) for movement because we use sets for floor tracking
- Request insertion is O(log n) if using priority queues (not strictly necessary)
- Dispatch is O(m) because we evaluate all elevators

## 🎯 Do's and Don'ts

### ✅ DO's
1. **Explain SCAN algorithm clearly**: Use disk scheduling analogy
2. **Use appropriate data structures**: Sets for floor tracking (O(1) lookup)
3. **Handle edge cases**: What if elevator is at requested floor?
4. **Consider direction**: Elevators moving toward request are better choices
5. **Add thread safety**: Use locks for concurrent access
6. **Track metrics**: Distance traveled, wait times, direction changes

### ❌ DON'Ts
1. **Don't use FCFS**: Inefficient - would create "crazy elevator" problem
2. **Don't ignore direction**: Choosing nearest elevator regardless of direction is suboptimal
3. **Don't forget edge cases**: Top/bottom floor, already at floor, no elevators available
4. **Don't overcomplicate initially**: Start with one elevator, then scale
5. **Don't hardcode values**: Use configuration objects
6. **Don't skip concurrency discussion**: Even if you don't fully implement

## 🎤 Expected Interview Questions & Answers

### Q1: "Why SCAN algorithm instead of FCFS (First-Come-First-Served)?"
**A**: "FCFS creates the 'crazy elevator' problem. Imagine requests for floors 1→20→2→19. The elevator would constantly change direction, which is:
- Inefficient (high travel distance)
- Poor user experience (long wait times)
- High wear on mechanical components

SCAN solves this by continuing in one direction until no more requests, serving all requests in the path. This:
- Minimizes direction changes
- Reduces average travel distance
- Prevents starvation (all requests eventually served)
- Matches user expectations (elevators continue in their direction)"

### Q2: "How do you decide which elevator should handle a request?"
**A**: "I use a scoring system with multiple factors:

1. **Distance** (weight: 1.0): Closer is better
2. **Direction alignment** (weight: 2.0): 
   - Elevator moving toward request floor = good
   - Elevator moving away = heavy penalty
   - Idle elevator = neutral
3. **Current load** (weight: 0.5): Prefer less loaded elevators

Lower score wins. Example:
- Elevator A: 5 floors away, moving toward request → score = 5
- Elevator B: 3 floors away, moving away → score = 3 + 20 = 23
- Choose A despite being farther because it's aligned"

### Q3: "How do you handle morning rush hour (everyone going up from lobby)?"
**A**: "Morning rush is a special case that breaks normal algorithms. Solutions:

1. **Batch Loading**: Fill elevators to capacity at lobby before departing
2. **Express Service**: Some elevators skip low floors, serve high floors only
3. **Predictive Positioning**: Stage idle elevators at lobby during rush hours
4. **Algorithm Switching**: Detect pattern, switch to rush-hour-optimized algorithm

Key insight: During rush, individual optimization doesn't matter - system throughput matters. Better to fill one elevator completely than send partially filled ones."

### Q4: "What if all elevators are busy when a request comes in?"
**A**: "Several strategies:

1. **Queue the request**: Add to pending queue, dispatch when elevator becomes available
2. **Timeout mechanism**: If request not served within X seconds, escalate or notify
3. **Dynamic addition**: If pending queue grows large, might need to add more elevators (long-term)
4. **Priority system**: Emergency/VIP requests could preempt normal requests

In my implementation, I use a pending queue with periodic retry logic. Request remains queued until an elevator becomes available."

### Q5: "How do you prevent all elevators from going to the same request?"
**A**: "This is a concurrency problem. Solution:

```python
class ElevatorController:
    def __init__(self):
        self.lock = RLock()  # System-wide lock
    
    def dispatch_request(self, request):
        with self.lock:  # Atomic operation
            # Select elevator
            elevator = self.strategy.select_elevator(self.elevators, request)
            
            # Immediately assign request to that elevator
            if elevator.add_request(request.floor):
                return elevator.id
            
            # Request dispatched - no other thread can select same elevator
            # for same request
```

The lock ensures dispatch + assignment is atomic. Once an elevator is assigned, other threads see its updated state (busy with this request)."

### Q6: "How would you scale this to a 100-story building?"
**A**: "100 stories requires architectural changes:

1. **Zone-based System**:
   - Divide into zones (1-30, 31-60, 61-90, 91-100)
   - Assign elevators to zones
   - Use sky lobbies for transfers

2. **Express + Local Elevators**:
   - Express: Only stop at major floors (1, 25, 50, 75, 100)
   - Local: Serve floors within zones

3. **Destination dispatch**:
   - User enters destination on ground floor
   - System assigns specific elevator
   - All passengers in elevator going to same zone

4. **Algorithm changes**:
   - Graph-based routing (elevator as nodes, floors as edges)
   - Multi-stage journey planning
   - Transfer optimization

Key insight: Single-bank approach doesn't scale beyond ~40 floors due to travel time and wait time."

### Q7: "What metrics would you track for monitoring?"
**A**: "Key metrics for elevator systems:

**User Experience**:
- Average wait time (button press → elevator arrival)
- Journey time (origin → destination)
- 95th/99th percentile wait times (outliers matter!)

**System Efficiency**:
- Average distance per trip
- Direction changes per hour
- Capacity utilization (% of max load)

**Operational**:
- Total trips per elevator
- Idle time percentage
- Energy consumption
- Maintenance alerts

**Business**:
- Peak hour handling (requests served during rush)
- SLA compliance (% requests served within target time)

Would implement a metrics collector that aggregates these in real-time for dashboards."

## 🧪 Testing Strategy

### Unit Tests to Write
```python
def test_scan_algorithm_basic():
    # Test SCAN visits floors in correct order
    elevator = Elevator(id=1, num_floors=10)
    elevator.add_request(5)
    elevator.add_request(3)
    elevator.add_request(7)
    
    # Starting at floor 1, should visit: 3 → 5 → 7 (all up)
    # Assert floor visit order

def test_dispatch_nearest_car():
    # Test nearest car strategy
    
def test_concurrent_requests():
    # Test thread safety
    
def test_peak_load():
    # Test system under heavy load

def test_edge_cases():
    # Already at floor, top/bottom floor, no available elevators
```

### Demo Scenarios
1. **Single Elevator**: Basic SCAN algorithm demonstration
2. **Multiple Elevators**: Dispatch strategy comparison
3. **Peak Load**: Morning rush simulation with 50+ requests
4. **Direction Changes**: Show SCAN minimizes direction reversals
5. **Concurrent Access**: Multiple threads making requests

## 🚀 Production Considerations

### Monitoring & Alerts
- Wait time exceeding SLA
- Elevator stuck detection
- High error rate
- Unusual usage patterns

### Optimization Strategies
- **Machine Learning**: Predict demand patterns, pre-position elevators
- **Dynamic Algorithm Selection**: Switch strategies based on time/load
- **Energy Optimization**: Minimize motor usage, regenerative braking
- **Maintenance Scheduling**: Predictive maintenance based on usage

### Safety & Reliability
- Emergency protocols (fire, earthquake)
- Graceful degradation (continue with fewer elevators)
- Redundancy (backup controller)
- Manual override capability

## 📖 Additional Learning Resources

### Algorithms
- "SCAN Disk Scheduling Algorithm" - OS textbooks
- "Look Algorithm" (variant of SCAN)
- "C-SCAN" (Circular SCAN for uniform service)

### Real Systems
- Otis Compass destination dispatch
- ThyssenKrupp TWIN elevators
- Kone UltraRope for high-rises

### Papers
- "Elevator Group Control Systems" - research papers
- "Optimization of Elevator Group Control" - AI approaches

---

## 💡 Final Interview Tips

### Communication Strategy
1. **Start with SCAN**: This is the core algorithm, explain it well
2. **Mention trade-offs**: SCAN vs FCFS vs Shortest-Seek-Time-First
3. **Scale incrementally**: One elevator → Multiple elevators → Dispatch → Concurrency
4. **Think aloud**: "Now I'm considering how to handle direction changes..."
5. **Ask for feedback**: "Does this approach make sense? Should I optimize for anything specific?"

### Common Pitfalls to Avoid
❌ Jumping to complex dispatch strategies before implementing basic SCAN
❌ Forgetting to handle idle elevators
❌ Not considering what happens when elevator reaches top/bottom floor
❌ Ignoring thread safety in multi-elevator system
❌ Using inefficient data structures (lists instead of sets for floor tracking)

### Time Management (45-minute interview)
- 0-5 min: Requirements and design discussion
- 5-25 min: Implement single elevator with SCAN
- 25-35 min: Add multi-elevator dispatch
- 35-40 min: Discuss scaling, edge cases
- 40-45 min: Testing, optimization discussion

### What Distinguishes Strong Candidates
✅ Immediately recognizes SCAN algorithm (shows OS knowledge)
✅ Explains why FCFS is bad before being asked
✅ Considers concurrency without prompting
✅ Brings up real-world scenarios (rush hour, emergencies)
✅ Discusses metrics and monitoring
✅ Knows when to stop optimizing and ship

Remember: Interviewers want to see problem-solving approach, not perfect code. Explain your thinking, discuss trade-offs, and demonstrate systematic thinking!
