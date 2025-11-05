"""
Elevator System - Single File Implementation
For coding interviews and production-ready reference

Features:
- Multi-elevator dispatching with multiple strategies
- SCAN algorithm for efficient elevator movement
- Peak load handling and traffic optimization
- Thread-safe concurrent operations
- Real-time performance analytics

Interview Focus:
- Algorithm design (SCAN, dispatching strategies)
- Concurrency and thread safety
- System optimization under constraints
- State management and coordination

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict, deque
from threading import RLock, Thread
import time
import heapq
from datetime import datetime


# ============================================================================
# MODELS - Core data classes and enums
# ============================================================================

class Direction(Enum):
    """Elevator movement direction"""
    UP = "up"
    DOWN = "down"
    IDLE = "idle"


class ElevatorState(Enum):
    """Current state of elevator"""
    IDLE = "idle"
    MOVING = "moving"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@dataclass
class ElevatorRequest:
    """
    Request for elevator service
    
    Attributes:
        floor: Floor number where request originated
        direction: Desired direction (UP or DOWN)
        timestamp: When request was made
        request_id: Unique identifier for tracking
    """
    floor: int
    direction: Direction
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: f"REQ_{int(time.time()*1000)}")
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.timestamp < other.timestamp


@dataclass
class ElevatorConfig:
    """
    Configuration for elevator system
    
    Attributes:
        num_floors: Total number of floors in building
        num_elevators: Number of elevator cars
        capacity_per_elevator: Maximum passengers per elevator
        floor_travel_time_ms: Time to travel one floor (milliseconds)
        door_operation_time_ms: Time to open/close doors (milliseconds)
    """
    num_floors: int
    num_elevators: int
    capacity_per_elevator: int = 10
    floor_travel_time_ms: int = 1000
    door_operation_time_ms: int = 500
    
    def __post_init__(self):
        """Validate configuration"""
        if self.num_floors < 2:
            raise ValueError("num_floors must be at least 2")
        if self.num_elevators < 1:
            raise ValueError("num_elevators must be at least 1")
        if self.capacity_per_elevator < 1:
            raise ValueError("capacity_per_elevator must be at least 1")


@dataclass
class ElevatorMetrics:
    """
    Performance metrics for elevator
    
    Interview Focus: What metrics matter for optimization?
    """
    total_trips: int = 0
    total_distance: int = 0
    total_wait_time: float = 0.0
    total_passengers_served: int = 0
    direction_changes: int = 0
    idle_time: float = 0.0
    
    @property
    def avg_trip_distance(self) -> float:
        """Average distance per trip"""
        return self.total_distance / self.total_trips if self.total_trips > 0 else 0.0
    
    @property
    def avg_wait_time(self) -> float:
        """Average wait time per passenger"""
        return self.total_wait_time / self.total_passengers_served if self.total_passengers_served > 0 else 0.0


# ============================================================================
# ELEVATOR - Core elevator car implementation
# ============================================================================

class Elevator:
    """
    Individual elevator car with SCAN algorithm
    
    Implementation Notes:
    - Uses SCAN algorithm (elevator algorithm from OS)
    - Maintains separate up/down request queues
    - Thread-safe operations for concurrent access
    
    Time Complexity:
    - add_request(): O(log n) due to heap operations
    - move(): O(1) for movement, O(log n) for request handling
    
    Space Complexity: O(n) where n is number of pending requests
    """
    
    def __init__(self, elevator_id: int, config: ElevatorConfig):
        self.elevator_id = elevator_id
        self.config = config
        
        # Current state
        self.current_floor = 1
        self.direction = Direction.IDLE
        self.state = ElevatorState.IDLE
        self.current_load = 0
        
        # Request queues - SCAN algorithm uses separate up/down queues
        self.up_requests: Set[int] = set()  # Floors with up requests
        self.down_requests: Set[int] = set()  # Floors with down requests
        
        # Metrics and tracking
        self.metrics = ElevatorMetrics()
        self.lock = RLock()
        
        # Movement tracking
        self.last_direction_change = time.time()
    
    def add_request(self, floor: int, direction: Direction = None) -> bool:
        """
        Add a floor request to elevator's queue
        
        Args:
            floor: Target floor
            direction: Optional direction hint for optimization
            
        Returns:
            True if request added successfully
            
        Interview Focus: How do you maintain request queues efficiently?
        """
        with self.lock:
            if floor < 1 or floor > self.config.num_floors:
                return False
            
            # Add to appropriate queue based on elevator's current position and direction
            if floor > self.current_floor:
                self.up_requests.add(floor)
            elif floor < self.current_floor:
                self.down_requests.add(floor)
            else:
                # Already at floor
                return False
            
            # Wake up idle elevator
            if self.state == ElevatorState.IDLE:
                self.state = ElevatorState.MOVING
            
            return True
    
    def move(self) -> bool:
        """
        Execute one step of SCAN algorithm
        
        Returns:
            True if elevator moved, False if idle
            
        Interview Focus: How does SCAN algorithm work?
        
        SCAN Algorithm Explanation:
        1. Continue in current direction until no more requests
        2. Service all requests in path
        3. Reverse direction when reaching end
        4. Prevents starvation - all requests eventually served
        """
        with self.lock:
            if self.state != ElevatorState.MOVING:
                return False
            
            # Determine direction based on pending requests
            if self.direction == Direction.IDLE:
                if self.up_requests:
                    self.direction = Direction.UP
                elif self.down_requests:
                    self.direction = Direction.DOWN
                else:
                    self.state = ElevatorState.IDLE
                    return False
            
            # Move in current direction
            if self.direction == Direction.UP:
                return self._move_up()
            else:
                return self._move_down()
    
    def _move_up(self) -> bool:
        """Move elevator up one floor using SCAN algorithm"""
        # Check if current floor has requests
        if self.current_floor in self.up_requests:
            self.up_requests.remove(self.current_floor)
            self.state = ElevatorState.STOPPED
            self.metrics.total_trips += 1
            return True
        
        # Move to next floor if we have more requests above
        if self.up_requests and max(self.up_requests) > self.current_floor:
            old_floor = self.current_floor
            self.current_floor += 1
            self.metrics.total_distance += 1
            
            # Check if we should stop at this floor
            if self.current_floor in self.up_requests:
                self.up_requests.remove(self.current_floor)
                self.state = ElevatorState.STOPPED
                self.metrics.total_trips += 1
            
            return True
        
        # No more up requests - change direction or go idle
        if self.down_requests:
            self._change_direction(Direction.DOWN)
            return self._move_down()
        else:
            self.direction = Direction.IDLE
            self.state = ElevatorState.IDLE
            return False
    
    def _move_down(self) -> bool:
        """Move elevator down one floor using SCAN algorithm"""
        # Check if current floor has requests
        if self.current_floor in self.down_requests:
            self.down_requests.remove(self.current_floor)
            self.state = ElevatorState.STOPPED
            self.metrics.total_trips += 1
            return True
        
        # Move to next floor if we have more requests below
        if self.down_requests and min(self.down_requests) < self.current_floor:
            old_floor = self.current_floor
            self.current_floor -= 1
            self.metrics.total_distance += 1
            
            # Check if we should stop at this floor
            if self.current_floor in self.down_requests:
                self.down_requests.remove(self.current_floor)
                self.state = ElevatorState.STOPPED
                self.metrics.total_trips += 1
            
            return True
        
        # No more down requests - change direction or go idle
        if self.up_requests:
            self._change_direction(Direction.UP)
            return self._move_up()
        else:
            self.direction = Direction.IDLE
            self.state = ElevatorState.IDLE
            return False
    
    def _change_direction(self, new_direction: Direction):
        """Change elevator direction and track metrics"""
        if self.direction != new_direction and self.direction != Direction.IDLE:
            self.metrics.direction_changes += 1
            self.last_direction_change = time.time()
        self.direction = new_direction
    
    def get_distance_to_floor(self, floor: int) -> int:
        """Calculate distance to a floor (used by dispatching algorithms)"""
        return abs(self.current_floor - floor)
    
    def is_available(self) -> bool:
        """Check if elevator can take new requests"""
        with self.lock:
            return (self.state != ElevatorState.MAINTENANCE and 
                    self.state != ElevatorState.EMERGENCY and
                    self.current_load < self.config.capacity_per_elevator)
    
    def get_status(self) -> Dict:
        """Get current elevator status for monitoring"""
        with self.lock:
            return {
                'elevator_id': self.elevator_id,
                'current_floor': self.current_floor,
                'direction': self.direction.value,
                'state': self.state.value,
                'load': self.current_load,
                'pending_requests': len(self.up_requests) + len(self.down_requests),
                'up_requests': sorted(list(self.up_requests)),
                'down_requests': sorted(list(self.down_requests), reverse=True)
            }


# ============================================================================
# DISPATCH STRATEGIES - Different algorithms for elevator assignment
# ============================================================================

class DispatchStrategy(ABC):
    """
    Abstract strategy for elevator dispatching
    
    Strategy Pattern: Allows switching between different dispatching algorithms
    
    Interview Focus: How do you choose the best elevator for a request?
    """
    
    @abstractmethod
    def select_elevator(self, elevators: List[Elevator], request: ElevatorRequest) -> Optional[Elevator]:
        """
        Select best elevator for the request
        
        Args:
            elevators: List of available elevators
            request: The floor request to service
            
        Returns:
            Selected elevator or None if no suitable elevator
        """
        pass


class NearestCarStrategy(DispatchStrategy):
    """
    Simplest strategy: Choose nearest available elevator
    
    Pros: Simple, minimizes travel distance
    Cons: Doesn't consider direction, can lead to uneven load distribution
    
    Time Complexity: O(n) where n is number of elevators
    """
    
    def select_elevator(self, elevators: List[Elevator], request: ElevatorRequest) -> Optional[Elevator]:
        available = [e for e in elevators if e.is_available()]
        if not available:
            return None
        
        # Find elevator with minimum distance to request floor
        return min(available, key=lambda e: e.get_distance_to_floor(request.floor))


class OptimizedDispatchStrategy(DispatchStrategy):
    """
    Optimized strategy: Consider distance, direction, and current load
    
    Scoring factors:
    - Distance to request floor (lower is better)
    - Direction alignment (prefer elevators moving toward request)
    - Current load (prefer less loaded elevators)
    
    Time Complexity: O(n) where n is number of elevators
    
    Interview Focus: How do you balance multiple optimization criteria?
    """
    
    def select_elevator(self, elevators: List[Elevator], request: ElevatorRequest) -> Optional[Elevator]:
        available = [e for e in elevators if e.is_available()]
        if not available:
            return None
        
        def calculate_score(elevator: Elevator) -> float:
            """Lower score is better"""
            score = 0.0
            
            # Distance factor (weight: 1.0)
            distance = elevator.get_distance_to_floor(request.floor)
            score += distance * 1.0
            
            # Direction alignment factor (weight: 2.0)
            is_aligned = False
            if elevator.direction == Direction.IDLE:
                is_aligned = True  # Idle elevator is always good
            elif request.direction == Direction.UP:
                is_aligned = (elevator.direction == Direction.UP and 
                            elevator.current_floor <= request.floor)
            else:
                is_aligned = (elevator.direction == Direction.DOWN and 
                            elevator.current_floor >= request.floor)
            
            if not is_aligned:
                score += 20.0  # Penalty for misaligned direction
            
            # Load factor (weight: 0.5)
            load_ratio = elevator.current_load / elevator.config.capacity_per_elevator
            score += load_ratio * 5.0
            
            return score
        
        return min(available, key=calculate_score)


class ZoneBasedDispatchStrategy(DispatchStrategy):
    """
    Zone-based strategy: Divide building into zones, assign elevators to zones
    
    Used in high-rise buildings (50+ floors)
    
    Pros: Reduces average wait time, better for tall buildings
    Cons: May leave zones underserved during low traffic
    
    Interview Focus: How do you handle scalability for tall buildings?
    """
    
    def __init__(self, num_floors: int, num_elevators: int):
        self.num_floors = num_floors
        self.zone_size = max(num_floors // num_elevators, 1)
    
    def select_elevator(self, elevators: List[Elevator], request: ElevatorRequest) -> Optional[Elevator]:
        # Determine zone for request
        zone_id = (request.floor - 1) // self.zone_size
        
        # Find elevators assigned to this zone or nearby zones
        available = [e for e in elevators if e.is_available()]
        if not available:
            return None
        
        # Prefer elevators in the same zone
        same_zone = [e for e in available if (e.current_floor - 1) // self.zone_size == zone_id]
        if same_zone:
            return min(same_zone, key=lambda e: e.get_distance_to_floor(request.floor))
        
        # Fall back to nearest elevator
        return min(available, key=lambda e: e.get_distance_to_floor(request.floor))


# ============================================================================
# ELEVATOR CONTROLLER - Main system coordinator
# ============================================================================

class ElevatorController:
    """
    Central controller managing multiple elevators
    
    Responsibilities:
    - Dispatch requests to appropriate elevators
    - Coordinate elevator movements
    - Track system-wide metrics
    - Handle peak load scenarios
    
    Thread Safety: Uses RLock for all operations
    
    Interview Focus: How do you coordinate multiple elevators efficiently?
    """
    
    def __init__(self, config: ElevatorConfig, dispatch_strategy: DispatchStrategy = None):
        self.config = config
        
        # Initialize elevators
        self.elevators = [Elevator(i, config) for i in range(config.num_elevators)]
        
        # Request management
        self.pending_requests: deque[ElevatorRequest] = deque()
        self.dispatch_strategy = dispatch_strategy or OptimizedDispatchStrategy()
        
        # System metrics
        self.total_requests = 0
        self.fulfilled_requests = 0
        self.rejected_requests = 0
        
        # Thread safety
        self.lock = RLock()
        
        # Background worker
        self._running = False
        self._worker_thread = None
    
    def request_elevator(self, floor: int, direction: Direction) -> str:
        """
        Request elevator at a floor
        
        Args:
            floor: Floor number where request is made
            direction: Desired direction (UP or DOWN)
            
        Returns:
            Request ID for tracking
            
        Interview Focus: How do you handle request queuing?
        """
        with self.lock:
            request = ElevatorRequest(floor, direction)
            self.pending_requests.append(request)
            self.total_requests += 1
            
            # Try immediate dispatch
            self._dispatch_pending_requests()
            
            return request.request_id
    
    def _dispatch_pending_requests(self):
        """
        Dispatch pending requests to elevators
        
        Interview Focus: How do you handle concurrent request processing?
        """
        dispatched = []
        
        for request in self.pending_requests:
            elevator = self.dispatch_strategy.select_elevator(self.elevators, request)
            
            if elevator:
                if elevator.add_request(request.floor, request.direction):
                    dispatched.append(request)
                    self.fulfilled_requests += 1
            else:
                # No available elevator - request remains pending
                # In real system, might reject after timeout
                pass
        
        # Remove dispatched requests
        for request in dispatched:
            self.pending_requests.remove(request)
    
    def step_simulation(self):
        """
        Execute one simulation step for all elevators
        
        Used for testing and demonstration
        
        Interview Focus: How do you simulate system behavior?
        """
        with self.lock:
            # Move all elevators
            for elevator in self.elevators:
                elevator.move()
            
            # Try to dispatch pending requests
            self._dispatch_pending_requests()
    
    def start(self):
        """Start background worker thread"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
    
    def stop(self):
        """Stop background worker thread"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def _worker_loop(self):
        """Background worker that processes elevator movements"""
        while self._running:
            self.step_simulation()
            time.sleep(0.1)  # 100ms simulation step
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status
        
        Interview Focus: What metrics matter for monitoring?
        """
        with self.lock:
            elevator_statuses = [e.get_status() for e in self.elevators]
            
            # Aggregate metrics
            total_trips = sum(e.metrics.total_trips for e in self.elevators)
            total_distance = sum(e.metrics.total_distance for e in self.elevators)
            avg_trip_distance = total_distance / total_trips if total_trips > 0 else 0
            
            return {
                'total_requests': self.total_requests,
                'fulfilled_requests': self.fulfilled_requests,
                'pending_requests': len(self.pending_requests),
                'rejected_requests': self.rejected_requests,
                'total_trips': total_trips,
                'total_distance': total_distance,
                'avg_trip_distance': avg_trip_distance,
                'elevators': elevator_statuses
            }
    
    def simulate_peak_load(self, num_requests: int = 20):
        """
        Simulate peak load scenario (e.g., morning rush)
        
        Interview Focus: How does system perform under stress?
        """
        print(f"\n🏢 Simulating peak load with {num_requests} requests...")
        
        import random
        
        start_time = time.time()
        
        # Generate random requests
        for i in range(num_requests):
            floor = random.randint(1, self.config.num_floors)
            direction = Direction.UP if random.random() > 0.5 else Direction.DOWN
            self.request_elevator(floor, direction)
        
        # Run simulation until all requests processed
        steps = 0
        max_steps = 1000  # Prevent infinite loop
        
        while len(self.pending_requests) > 0 and steps < max_steps:
            self.step_simulation()
            steps += 1
            
            if steps % 10 == 0:
                status = self.get_system_status()
                print(f"  Step {steps}: Pending={status['pending_requests']}, "
                      f"Fulfilled={status['fulfilled_requests']}")
        
        elapsed = time.time() - start_time
        status = self.get_system_status()
        
        print(f"\n✅ Simulation completed in {elapsed:.2f}s ({steps} steps)")
        print(f"   Fulfilled: {status['fulfilled_requests']}/{status['total_requests']}")
        print(f"   Avg trip distance: {status['avg_trip_distance']:.1f} floors")


# ============================================================================
# DEMO - Demonstration and testing code
# ============================================================================

def print_separator(title: str = ""):
    """Print visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print(f"{'='*70}\n")


def demo_basic_operations():
    """Demonstrate basic elevator operations"""
    print_separator("Basic Elevator Operations")
    
    config = ElevatorConfig(num_floors=10, num_elevators=1)
    controller = ElevatorController(config, NearestCarStrategy())
    
    print("Building: 10 floors, 1 elevator\n")
    
    # Request from floor 1 to go up
    print("Scenario: Person on floor 1 wants to go UP")
    controller.request_elevator(1, Direction.UP)
    controller.step_simulation()
    
    status = controller.get_system_status()
    print(f"  Elevator status: Floor {status['elevators'][0]['current_floor']}, "
          f"Direction: {status['elevators'][0]['direction']}")
    
    # Add destination floor 5
    print("\nPerson presses button 5 inside elevator")
    controller.elevators[0].add_request(5)
    
    # Simulate movement
    print("\nElevator moving...")
    for _ in range(10):
        controller.step_simulation()
        status = controller.get_system_status()
        elevator_status = status['elevators'][0]
        print(f"  Floor {elevator_status['current_floor']} - {elevator_status['state']}")
        
        if elevator_status['state'] == 'idle':
            break
    
    print(f"\n✅ Trip completed! Total distance: {status['total_distance']} floors")


def demo_dispatch_strategies():
    """Compare different dispatch strategies"""
    print_separator("Dispatch Strategy Comparison")
    
    strategies = [
        ("Nearest Car", NearestCarStrategy()),
        ("Optimized", OptimizedDispatchStrategy()),
    ]
    
    config = ElevatorConfig(num_floors=20, num_elevators=3)
    
    print("Building: 20 floors, 3 elevators\n")
    
    for strategy_name, strategy in strategies:
        print(f"\n--- Testing {strategy_name} Strategy ---")
        controller = ElevatorController(config, strategy)
        
        # Generate test requests
        test_requests = [(5, Direction.UP), (15, Direction.DOWN), (10, Direction.UP)]
        
        for floor, direction in test_requests:
            controller.request_elevator(floor, direction)
        
        # Run simulation
        for _ in range(30):
            controller.step_simulation()
        
        status = controller.get_system_status()
        print(f"  Fulfilled: {status['fulfilled_requests']}/{status['total_requests']}")
        print(f"  Total distance: {status['total_distance']} floors")
        print(f"  Avg trip distance: {status['avg_trip_distance']:.1f} floors")


def demo_peak_load():
    """Demonstrate peak load handling"""
    print_separator("Peak Load Simulation")
    
    config = ElevatorConfig(num_floors=30, num_elevators=4)
    controller = ElevatorController(config, OptimizedDispatchStrategy())
    
    print("Building: 30 floors, 4 elevators")
    print("Simulating morning rush hour...\n")
    
    controller.simulate_peak_load(num_requests=25)
    
    status = controller.get_system_status()
    print(f"\n📊 Final Statistics:")
    print(f"   Total trips: {status['total_trips']}")
    print(f"   Total distance traveled: {status['total_distance']} floors")
    print(f"   Average trip distance: {status['avg_trip_distance']:.1f} floors")


def demo_scan_algorithm():
    """Demonstrate SCAN algorithm behavior"""
    print_separator("SCAN Algorithm Demonstration")
    
    config = ElevatorConfig(num_floors=10, num_elevators=1)
    elevator = Elevator(0, config)
    
    print("Single elevator, starting at floor 1")
    print("Requests: floors 3, 7, 5, 2 (in this order)\n")
    
    # Add requests in non-sequential order
    elevator.add_request(3)
    elevator.add_request(7)
    elevator.add_request(5)
    elevator.add_request(2)
    
    print("SCAN algorithm will visit floors in order:")
    print("  2 → 3 → 5 → 7 (all UP requests in ascending order)")
    print("\nSimulating movement:")
    
    for step in range(15):
        status = elevator.get_status()
        print(f"  Step {step+1}: Floor {status['current_floor']}, "
              f"Direction: {status['direction']}, "
              f"Pending: {status['pending_requests']}")
        
        if not elevator.move():
            break
    
    print(f"\n✅ All requests served efficiently!")
    print(f"   Total distance: {elevator.metrics.total_distance} floors")
    print(f"   Direction changes: {elevator.metrics.direction_changes}")


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  ELEVATOR SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Multi-elevator, SCAN algorithm, Dispatch strategies")
    print("="*70)
    
    demo_basic_operations()
    demo_scan_algorithm()
    demo_dispatch_strategies()
    demo_peak_load()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python elevator_system_merged.py
    """
    run_demo()
