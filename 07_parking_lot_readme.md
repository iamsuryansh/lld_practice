# Parking Lot System - Interview Guide

## 🎯 System Overview

**Objective**: Design a comprehensive parking lot management system that handles multi-level parking, different vehicle types, spot allocation strategies, and payment processing.

**Core Components**:
- Multi-level parking structure with different spot types
- Vehicle management (Car, Bike, Truck)
- Smart allocation strategies
- Payment processing system
- Real-time availability tracking
- Thread-safe concurrent operations

---

## 📋 Step-by-Step Implementation Guide

### Phase 1: Define Core Entities
```
1. Vehicle Types & Spot Types
   - Define enums for vehicle types (BIKE, CAR, TRUCK)
   - Define spot types (BIKE_SPOT, COMPACT_SPOT, REGULAR_SPOT, LARGE_SPOT)
   - Create compatibility matrix

2. Core Data Models
   - Vehicle class with license plate, type, owner info
   - ParkingSpot class with status, type, floor, section
   - ParkingTicket class with entry/exit times, cost
   - Payment class with amount, method, status
```

### Phase 2: Design Allocation Strategies
```
1. Strategy Pattern Implementation
   - Abstract SpotAllocationStrategy base class
   - ClosestToEntranceStrategy (minimize walking)
   - OptimalFitStrategy (maximize space utilization)
   - Allow runtime strategy switching

2. Spot Compatibility Logic
   - Define which vehicles can use which spots
   - Implement size-based allocation rules
   - Handle edge cases (truck needs large spot)
```

### Phase 3: Payment Processing
```
1. Payment Methods Support
   - Cash, Credit Card, Mobile Payment
   - Abstract PaymentProcessor interface
   - Payment status tracking (PENDING, COMPLETED, FAILED)

2. Pricing Strategies
   - HourlyPricingStrategy with different rates
   - FlatRatePricingStrategy for simplicity
   - Dynamic pricing capabilities
```

### Phase 4: Main System Implementation
```
1. ParkingLotSystem Class
   - Thread-safe operations with RLock
   - Spot management and availability tracking
   - Ticket lifecycle management
   - Statistics and reporting

2. Advanced Features
   - Spot reservation system
   - Parking history tracking
   - Real-time availability by vehicle type
   - System statistics and analytics
```

---

## 🚀 Key Design Decisions & Rationale

### 1. **Strategy Pattern for Allocation**
- **Why**: Different allocation needs (closest vs optimal fit)
- **Benefit**: Runtime strategy switching, extensible design
- **Implementation**: Abstract base class with concrete strategies

### 2. **Enum-Based Type Safety**
- **Why**: Prevent invalid vehicle/spot type combinations
- **Benefit**: Compile-time safety, clear intent
- **Implementation**: VehicleType, SpotType, PaymentMethod enums

### 3. **Dataclass for Data Models**
- **Why**: Reduced boilerplate, automatic __eq__ and __hash__
- **Benefit**: Clean, readable code with built-in functionality
- **Implementation**: Vehicle, ParkingSpot, Payment dataclasses

### 4. **Thread-Safe Design**
- **Why**: Concurrent parking/exit operations in real system
- **Benefit**: Data consistency, prevents race conditions
- **Implementation**: RLock for all critical sections

---

## 💡 Interview Dos and Don'ts

### ✅ DO:
- **Start with entities**: Clearly define Vehicle, ParkingSpot, Ticket classes
- **Use Strategy pattern**: For allocation and pricing flexibility
- **Handle edge cases**: What if no spots available? Payment fails?
- **Consider scalability**: How system handles 10,000 spots?
- **Thread safety**: Discuss concurrent operations early
- **Clear state transitions**: AVAILABLE → OCCUPIED → AVAILABLE
- **Extensible design**: Easy to add new vehicle types/strategies

### ❌ DON'T:
- **Hardcode pricing**: Use strategy pattern for flexibility
- **Ignore concurrency**: Real systems have concurrent users
- **Forget reservations**: Important for user experience
- **Skip error handling**: Payment failures, invalid operations
- **Overcomplicate initially**: Start simple, add features iteratively
- **Mix responsibilities**: Keep allocation, pricing, payment separate
- **Forget statistics**: Important for business operations

---

## 🎭 Common Interview Scenarios

### Scenario 1: "How do you handle peak hours?"
```python
# Strategy: Implement priority-based allocation
class PeakHourStrategy(SpotAllocationStrategy):
    def find_spot(self, vehicle_type, available_spots):
        # Reserve some spots for different vehicle types
        # Implement dynamic pricing during peak hours
        pass
```

### Scenario 2: "What if payment system is down?"
```python
# Strategy: Graceful degradation
def process_payment(self, ticket_id, method, details):
    try:
        payment = self.payment_processor.process_payment(amount, method, details)
        if payment.status == PaymentStatus.FAILED:
            # Allow exit with pending payment, send notification
            self._mark_payment_pending(ticket)
        return payment
    except PaymentSystemException:
        # Fallback to cash or manual processing
        return self._handle_payment_system_down(ticket)
```

### Scenario 3: "How do you prevent double booking?"
```python
# Strategy: Atomic operations with locking
def park_vehicle(self, vehicle):
    with self.lock:  # Atomic operation
        if vehicle in self.vehicle_to_ticket:
            return None  # Already parked
        
        spot = self.find_available_spot(vehicle.vehicle_type)
        if spot:
            spot.status = ParkingSpotStatus.OCCUPIED
            # Create ticket and update mappings atomically
```

---

## ❓ Interview Questions & Detailed Answers

### **Q1: Walk me through your class design for the parking lot system.**

**A1**: "I'll design this using object-oriented principles with clear separation of concerns:

**Core Entities:**
- `Vehicle`: Represents cars, bikes, trucks with license plate and owner info
- `ParkingSpot`: Tracks location, type, status, and current vehicle
- `ParkingTicket`: Links vehicle to spot with timing and payment info
- `Payment`: Handles payment methods, amounts, and transaction status

**Design Patterns:**
- **Strategy Pattern**: For allocation algorithms (closest vs optimal fit)
- **Strategy Pattern**: For pricing (hourly vs flat rate)
- **Abstract Factory**: For payment processing methods

**Key Relationships:**
- Vehicle ↔ ParkingTicket (1:1 active, 1:N historical)
- ParkingSpot ↔ ParkingTicket (1:1 when occupied)
- ParkingTicket ↔ Payment (1:1 when paid)

This design allows for flexibility in allocation strategies and pricing models while maintaining clean separation of concerns."

### **Q2: How do you handle different vehicle types and spot allocation?**

**A2**: "I use a compatibility matrix approach with smart allocation strategies:

**Spot Compatibility:**
```python
SPOT_COMPATIBILITY = {
    SpotType.BIKE_SPOT: [VehicleType.BIKE],
    SpotType.COMPACT_SPOT: [VehicleType.BIKE, VehicleType.CAR],
    SpotType.REGULAR_SPOT: [VehicleType.BIKE, VehicleType.CAR],
    SpotType.LARGE_SPOT: [VehicleType.BIKE, VehicleType.CAR, VehicleType.TRUCK]
}
```

**Allocation Strategies:**
1. **OptimalFitStrategy**: Assigns smallest suitable spot to maximize utilization
2. **ClosestToEntranceStrategy**: Minimizes walking distance for customers

**Benefits:**
- Prevents trucks from taking compact spots unnecessarily
- Maximizes space utilization
- Easy to add new vehicle types or spots
- Runtime strategy switching based on conditions"

### **Q3: How do you ensure thread safety in concurrent operations?**

**A3**: "I implement comprehensive thread safety using multiple techniques:

**Primary Mechanism:**
- Use `threading.RLock()` for reentrant locking
- All critical operations (park, exit, payment) are atomic

**Critical Sections:**
```python
def park_vehicle(self, vehicle):
    with self.lock:  # Entire operation is atomic
        if vehicle in self.vehicle_to_ticket:
            return None  # Prevent double parking
        
        spot = self.find_available_spot(vehicle.vehicle_type)
        if spot:
            # Atomically update spot status and create ticket
            spot.status = ParkingSpotStatus.OCCUPIED
            ticket = self.create_ticket(vehicle, spot)
            self.update_mappings(vehicle, ticket)
```

**Consistency Guarantees:**
- No race conditions between spot checking and allocation
- Vehicle-to-ticket mapping always consistent
- Statistics updates are atomic
- Payment processing is transactional

**Testing:** The system includes concurrent operation tests with multiple threads to verify thread safety."

### **Q4: How would you handle payment processing and failures?**

**A4**: "I implement a robust payment system with failure handling:

**Payment Architecture:**
```python
class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount, method, details) -> Payment:
        pass

class SimplePaymentProcessor(PaymentProcessor):
    def process_payment(self, amount, method, details):
        # Simulate real payment gateway integration
        payment = Payment(id, amount, method, status=PENDING)
        
        try:
            # Call external payment service
            result = self.payment_gateway.charge(amount, details)
            payment.status = COMPLETED if result.success else FAILED
        except PaymentServiceException:
            payment.status = FAILED
        
        return payment
```

**Failure Handling Strategies:**
1. **Graceful Degradation**: Allow exit with pending payment, collect later
2. **Retry Logic**: Automatic retry with exponential backoff
3. **Fallback Methods**: If credit card fails, offer cash/mobile payment
4. **Manual Override**: Admin can manually process stuck payments

**State Management:**
- Payments are tracked separately from parking sessions
- Failed payments can be retried without re-parking
- Refund capability for overpayments or disputes"

### **Q5: How do you implement spot reservation functionality?**

**A5**: "Spot reservation adds another layer of complexity with time-based logic:

**Reservation Logic:**
```python
def reserve_spot(self, vehicle_type, duration_minutes=30):
    with self.lock:
        available_spots = self.get_available_spots()
        spot = self.allocation_strategy.find_spot(vehicle_type, available_spots)
        
        if spot:
            spot.status = ParkingSpotStatus.RESERVED
            spot.reserved_until = datetime.now() + timedelta(minutes=duration_minutes)
            return spot.spot_id
        return None

def is_available(self, spot):
    if spot.status == ParkingSpotStatus.AVAILABLE:
        return True
    if spot.status == ParkingSpotStatus.RESERVED:
        return datetime.now() > spot.reserved_until  # Expired reservation
    return False
```

**Key Features:**
- Time-based automatic expiration
- Prevents reservation abuse (limited duration)
- Seamless integration with regular allocation
- Can be extended for premium users or advance booking

**Business Logic:**
- Reservations expire automatically
- Reserved spots become available if not claimed
- Priority system can be added (premium vs regular users)"

### **Q6: How would you scale this system for a large parking facility?**

**A6**: "Scaling requires both architectural and implementation considerations:

**Database Layer:**
- Replace in-memory storage with persistent database
- Use connection pooling for concurrent access
- Implement proper indexing (license_plate, spot_id, timestamps)

**Caching Strategy:**
```python
class CachedAvailabilityManager:
    def __init__(self):
        self.availability_cache = {}  # vehicle_type -> count
        self.cache_expiry = 30  # seconds
    
    def get_availability(self, vehicle_type):
        if self.is_cache_valid():
            return self.availability_cache[vehicle_type]
        
        # Refresh cache from database
        self.refresh_cache()
        return self.availability_cache[vehicle_type]
```

**Distributed Architecture:**
- Microservices: Parking, Payment, Analytics as separate services
- Message queues for async operations (payment processing, notifications)
- Load balancers for high availability

**Performance Optimizations:**
- Batch operations for statistics updates
- Async payment processing
- Spot allocation algorithms with O(log n) complexity
- Regional sharding for very large facilities"

### **Q7: How do you handle different pricing models?**

**A7**: "I use the Strategy pattern for flexible pricing with business rule support:

**Pricing Strategies:**
```python
class HourlyPricingStrategy(PricingStrategy):
    def __init__(self):
        self.base_rates = {VehicleType.BIKE: 2.0, VehicleType.CAR: 5.0}
        self.peak_multiplier = 1.5
        self.peak_hours = [(8, 10), (17, 19)]  # Rush hours
    
    def calculate_cost(self, vehicle_type, duration, entry_time):
        base_cost = self.get_hourly_cost(vehicle_type, duration)
        
        if self.is_peak_hour(entry_time):
            return base_cost * self.peak_multiplier
        
        return base_cost

class DynamicPricingStrategy(PricingStrategy):
    def calculate_cost(self, vehicle_type, duration, occupancy_rate):
        base_cost = self.get_base_cost(vehicle_type, duration)
        
        # Increase price based on occupancy
        if occupancy_rate > 0.8:
            return base_cost * 1.3
        elif occupancy_rate > 0.6:
            return base_cost * 1.1
        
        return base_cost
```

**Advanced Features:**
- Peak hour multipliers
- Occupancy-based dynamic pricing
- Loyalty program discounts
- Subscription-based monthly parking
- Event-based surge pricing

**Configuration:**
- Pricing rules stored in database
- Runtime pricing strategy switching
- A/B testing capability for pricing experiments"

### **Q8: How do you implement parking analytics and reporting?**

**A8**: "Analytics are crucial for business operations and optimization:

**Statistics Collection:**
```python
class ParkingAnalytics:
    def __init__(self):
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        self.hourly_occupancy = {}
        self.revenue_tracking = defaultdict(float)
    
    def record_parking_event(self, event_type, vehicle_type, timestamp, revenue=0):
        date = timestamp.date()
        hour = timestamp.hour
        
        self.daily_stats[date][f'{event_type}_{vehicle_type}'] += 1
        self.hourly_occupancy[(date, hour)] = self.calculate_occupancy()
        
        if revenue > 0:
            self.revenue_tracking[date] += revenue
    
    def generate_business_report(self, start_date, end_date):
        return {
            'total_revenue': sum(self.revenue_tracking.values()),
            'average_occupancy': self.calculate_average_occupancy(),
            'peak_hours': self.identify_peak_hours(),
            'vehicle_type_distribution': self.get_vehicle_distribution(),
            'popular_floors': self.get_floor_popularity()
        }
```

**Key Metrics:**
- Occupancy rates by time/floor/section
- Revenue per vehicle type
- Average parking duration
- Peak usage patterns
- Spot utilization efficiency

**Business Intelligence:**
- Identify underutilized areas
- Optimize pricing based on demand patterns
- Predict maintenance needs based on usage
- Staff scheduling optimization"

### **Q9: How would you handle system failures and recovery?**

**A9**: "Robust failure handling is essential for a production parking system:

**Data Persistence:**
```python
class PersistentParkingLotSystem(ParkingLotSystem):
    def __init__(self, name, total_floors, db_connection):
        super().__init__(name, total_floors)
        self.db = db_connection
        self.auto_save_interval = 30  # seconds
        self._start_auto_save()
    
    def park_vehicle(self, vehicle):
        with self.lock:
            ticket = super().park_vehicle(vehicle)
            if ticket:
                self.db.save_ticket(ticket)  # Immediate persistence
                self.db.update_spot_status(ticket.spot.spot_id, 'OCCUPIED')
            return ticket
    
    def recover_from_failure(self):
        # Reload active tickets from database
        active_tickets = self.db.get_active_tickets()
        for ticket_data in active_tickets:
            self.restore_ticket_state(ticket_data)
```

**Failure Scenarios:**
1. **System Crash**: Restore state from database on restart
2. **Payment Gateway Down**: Queue payments for later processing
3. **Database Connection Lost**: Cache operations and sync when restored
4. **Hardware Failure**: Failover to backup systems

**Recovery Mechanisms:**
- Transaction logs for operation replay
- Checkpoint-based state snapning
- Distributed system with multiple nodes
- Manual admin override capabilities

**Monitoring:**
- Health checks for all system components
- Automated alerts for system anomalies
- Graceful degradation with reduced functionality"

### **Q10: How do you optimize spot allocation algorithms for performance?**

**A10**: "Optimization focuses on reducing algorithm complexity and smart data structures:

**Efficient Data Structures:**
```python
class OptimizedSpotManager:
    def __init__(self):
        # Group spots by type and floor for faster lookup
        self.spots_by_type = {
            spot_type: defaultdict(list) 
            for spot_type in SpotType
        }
        
        # Priority queues for different allocation strategies
        self.closest_spots = {
            vehicle_type: []  # Min-heap by floor number
            for vehicle_type in VehicleType
        }
    
    def find_optimal_spot(self, vehicle_type):
        # O(log n) lookup instead of O(n) linear search
        suitable_spot_types = self.get_compatible_spots(vehicle_type)
        
        for spot_type in suitable_spot_types:
            if self.closest_spots[vehicle_type]:
                return heapq.heappop(self.closest_spots[vehicle_type])
        
        return None
    
    def release_spot(self, spot):
        # O(log n) insertion back into priority queue
        for vehicle_type in self.get_compatible_vehicles(spot.spot_type):
            heapq.heappush(
                self.closest_spots[vehicle_type],
                (spot.floor_number, spot.spot_id, spot)
            )
```

**Algorithm Optimizations:**
1. **Pre-sorted Structures**: Maintain spots sorted by allocation criteria
2. **Lazy Evaluation**: Only compute allocations when needed
3. **Batch Operations**: Process multiple operations together
4. **Caching**: Cache allocation results for similar requests

**Performance Metrics:**
- Spot allocation: O(log n) instead of O(n)
- Availability lookup: O(1) with proper indexing
- Statistics computation: O(1) with incremental updates
- Memory usage: O(n) with efficient data structures"

### **Q11: How would you implement a mobile app integration?**

**A11**: "Mobile integration requires a well-designed API and real-time capabilities:

**RESTful API Design:**
```python
from flask import Flask, jsonify, request

class ParkingLotAPI:
    def __init__(self, parking_system):
        self.parking_system = parking_system
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/api/availability', methods=['GET'])
        def get_availability():
            vehicle_type = request.args.get('vehicle_type', 'CAR')
            availability = self.parking_system.get_availability_by_type()
            return jsonify(availability)
        
        @self.app.route('/api/park', methods=['POST'])
        def park_vehicle():
            data = request.json
            vehicle = Vehicle(
                data['license_plate'],
                VehicleType(data['vehicle_type']),
                data['owner_name']
            )
            
            ticket = self.parking_system.park_vehicle(vehicle)
            if ticket:
                return jsonify({
                    'success': True,
                    'ticket_id': ticket.ticket_id,
                    'spot_id': ticket.spot.spot_id,
                    'floor': ticket.spot.floor_number
                })
            
            return jsonify({'success': False, 'error': 'No spots available'})
```

**Real-time Features:**
- WebSocket connections for live availability updates
- Push notifications for parking expiration warnings
- QR code generation for quick entry/exit
- GPS integration for spot navigation

**Mobile-Specific Features:**
- Spot reservation with countdown timer
- Mobile payment integration (Apple Pay, Google Pay)
- Digital parking tickets
- Parking history and receipts
- Customer support chat integration"

### **Q12: How do you handle security and access control?**

**A12**: "Security is multi-layered covering authentication, authorization, and data protection:

**Authentication System:**
```python
class SecurityManager:
    def __init__(self):
        self.user_sessions = {}
        self.admin_users = set()
        self.access_logs = []
    
    def authenticate_user(self, username, password):
        # Implement secure authentication
        hashed_password = self.hash_password(password)
        if self.verify_credentials(username, hashed_password):
            session_token = self.generate_session_token()
            self.user_sessions[session_token] = {
                'username': username,
                'login_time': datetime.now(),
                'permissions': self.get_user_permissions(username)
            }
            return session_token
        return None
    
    def authorize_operation(self, session_token, operation):
        session = self.user_sessions.get(session_token)
        if not session:
            return False
        
        required_permission = self.get_required_permission(operation)
        return required_permission in session['permissions']
```

**Access Control Levels:**
1. **Public**: Availability checking, spot reservation
2. **Authenticated Users**: Parking, payment, history
3. **Staff**: Manual overrides, customer support
4. **Admins**: System configuration, analytics, user management

**Data Security:**
- Encrypt sensitive data (payment info, personal details)
- Secure API endpoints with rate limiting
- Audit logs for all system operations
- GDPR compliance for personal data handling
- PCI compliance for payment processing

**Physical Security:**
- Integration with gate controllers
- Camera system integration for license plate recognition
- RFID/NFC support for access cards
- Emergency override mechanisms"

### **Q13: How would you implement license plate recognition integration?**

**A13**: "License plate recognition adds automation and security to the system:

**LPR Integration Architecture:**
```python
class LicensePlateRecognitionSystem:
    def __init__(self, parking_system):
        self.parking_system = parking_system
        self.confidence_threshold = 0.85
        self.manual_verification_queue = []
    
    def process_entry_gate(self, camera_image):
        # Integrate with LPR service (AWS Rekognition, OpenALPR, etc.)
        lpr_result = self.lpr_service.recognize_plate(camera_image)
        
        if lpr_result.confidence > self.confidence_threshold:
            license_plate = lpr_result.plate_number
            
            # Check if vehicle is already in system
            existing_vehicle = self.find_vehicle_by_plate(license_plate)
            if existing_vehicle:
                return self.process_return_customer(existing_vehicle)
            else:
                return self.process_new_customer(license_plate, camera_image)
        else:
            # Queue for manual verification
            self.manual_verification_queue.append({
                'image': camera_image,
                'timestamp': datetime.now(),
                'raw_result': lpr_result
            })
            return self.generate_temporary_ticket()
    
    def process_exit_gate(self, license_plate):
        active_ticket = self.parking_system.find_active_ticket(license_plate)
        if active_ticket and active_ticket.is_paid():
            self.open_exit_gate()
            return {'success': True, 'message': 'Have a nice day!'}
        elif active_ticket:
            return {
                'success': False, 
                'message': 'Payment required',
                'amount': active_ticket.total_cost
            }
        else:
            return {'success': False, 'message': 'No active parking session'}
```

**Advanced Features:**
- Multiple camera angles for better accuracy
- Machine learning model training with facility-specific data
- Integration with blacklist/whitelist databases
- Automated incident detection (wrong-way driving, tailgating)

**Fallback Mechanisms:**
- Manual ticket dispensing for LPR failures
- Staff override capabilities
- Mobile app-based entry for registered users
- RFID backup system for regular customers"

### **Q14: How do you implement dynamic pricing based on demand?**

**A14**: "Dynamic pricing optimizes revenue and space utilization based on real-time demand:

**Dynamic Pricing Engine:**
```python
class DynamicPricingEngine(PricingStrategy):
    def __init__(self):
        self.base_rates = {VehicleType.CAR: 5.0, VehicleType.BIKE: 2.0}
        self.demand_multipliers = {
            (0.0, 0.5): 0.8,    # Low demand - discount
            (0.5, 0.7): 1.0,    # Normal demand - base rate
            (0.7, 0.85): 1.3,   # High demand - premium
            (0.85, 1.0): 1.8    # Very high demand - surge pricing
        }
        
    def calculate_cost(self, vehicle_type, duration, context):
        base_cost = self.get_base_cost(vehicle_type, duration)
        
        # Get current demand factors
        occupancy_rate = context['occupancy_rate']
        time_of_day = context['time_of_day']
        day_of_week = context['day_of_week']
        special_event = context.get('special_event', False)
        
        # Apply demand multiplier
        demand_multiplier = self.get_demand_multiplier(occupancy_rate)
        
        # Apply time-based adjustments
        time_multiplier = self.get_time_multiplier(time_of_day, day_of_week)
        
        # Apply event-based surge pricing
        event_multiplier = 2.0 if special_event else 1.0
        
        final_cost = base_cost * demand_multiplier * time_multiplier * event_multiplier
        return round(final_cost, 2)
    
    def predict_demand(self, timestamp):
        # Use historical data and machine learning for demand prediction
        historical_occupancy = self.get_historical_data(timestamp)
        weather_factor = self.get_weather_impact(timestamp)
        event_factor = self.check_nearby_events(timestamp)
        
        predicted_demand = self.ml_model.predict([
            historical_occupancy, weather_factor, event_factor
        ])
        
        return predicted_demand
```

**Demand Factors:**
- **Current Occupancy**: Real-time space availability
- **Historical Patterns**: Past usage data for similar times
- **Weather Conditions**: Impact on driving vs public transport
- **Local Events**: Concerts, sports games, conferences
- **Economic Indicators**: Local economic activity levels

**Implementation Considerations:**
- Price change notifications to users
- Maximum price limits to prevent customer alienation
- Gradual price adjustments to avoid shock
- Competitor price monitoring
- Revenue optimization algorithms"

### **Q15: How would you handle multi-tenant parking lots (shared facilities)?**

**A15**: "Multi-tenant facilities require complex resource allocation and billing:

**Tenant Management System:**
```python
class MultiTenantParkingSystem(ParkingLotSystem):
    def __init__(self, name, total_floors):
        super().__init__(name, total_floors)
        self.tenants = {}
        self.spot_allocations = {}  # tenant_id -> allocated spots
        self.tenant_usage = defaultdict(lambda: defaultdict(int))
    
    def add_tenant(self, tenant_id, name, allocated_spots, pricing_plan):
        self.tenants[tenant_id] = {
            'name': name,
            'allocated_spots': allocated_spots,
            'pricing_plan': pricing_plan,
            'monthly_allowance': pricing_plan.get('monthly_spots', 0),
            'overflow_rate': pricing_plan.get('overflow_rate', 1.5)
        }
        
        # Reserve spots for tenant
        self.allocate_spots_to_tenant(tenant_id, allocated_spots)
    
    def park_vehicle_for_tenant(self, vehicle, tenant_id):
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
        
        # Check tenant's allocated spots first
        tenant_spots = self.get_tenant_spots(tenant_id)
        available_tenant_spots = [s for s in tenant_spots if s.is_available()]
        
        if available_tenant_spots:
            # Use tenant's allocated spot
            spot = self.allocation_strategy.find_spot(
                vehicle.vehicle_type, available_tenant_spots
            )
            pricing = tenant['pricing_plan']['regular_rate']
        else:
            # Use overflow parking if allowed
            if tenant['pricing_plan'].get('allow_overflow', False):
                overflow_spots = self.get_overflow_spots()
                spot = self.allocation_strategy.find_spot(
                    vehicle.vehicle_type, overflow_spots
                )
                pricing = tenant['pricing_plan']['overflow_rate']
            else:
                return None  # No overflow allowed
        
        if spot:
            ticket = self.create_ticket(vehicle, spot, tenant_id, pricing)
            self.tenant_usage[tenant_id]['monthly_usage'] += 1
            return ticket
        
        return None
```

**Tenant Models:**
1. **Reserved Allocation**: Fixed number of dedicated spots
2. **Shared Pool**: Guaranteed minimum with overflow access
3. **Usage-Based**: Pay per use with volume discounts
4. **Hybrid**: Combination of reserved and shared access

**Billing Integration:**
- Monthly usage reports per tenant
- Different pricing tiers and discount structures
- Automated billing and invoice generation
- Usage analytics and optimization recommendations

**Management Features:**
- Real-time tenant usage monitoring
- Spot reallocation based on usage patterns
- Tenant-specific reporting and analytics
- SLA monitoring and compliance reporting"

---

## 🔧 Advanced Scenarios & Extensions

### **Scenario: Integration with Smart City Infrastructure**

```python
class SmartCityParkingHub:
    def __init__(self):
        self.connected_lots = {}
        self.traffic_management = TrafficManagementSystem()
        self.public_transport = PublicTransportAPI()
    
    def find_optimal_parking(self, user_location, destination, preferences):
        # Consider traffic, walking distance, pricing, availability
        nearby_lots = self.find_lots_near_destination(destination)
        
        recommendations = []
        for lot in nearby_lots:
            score = self.calculate_lot_score(
                user_location, lot, destination, preferences
            )
            recommendations.append((lot, score))
        
        # Sort by score and return top recommendations
        return sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    def integrate_with_navigation(self, parking_choice, user_location):
        # Provide turn-by-turn navigation to parking lot
        # Reserve spot during navigation
        # Notify of traffic updates affecting parking choice
        pass
```

### **Scenario: Environmental and Sustainability Features**

```python
class SustainableParkingSystem(ParkingLotSystem):
    def __init__(self, name, total_floors):
        super().__init__(name, total_floors)
        self.ev_charging_spots = []
        self.solar_panel_system = SolarPanelManager()
        self.carbon_tracking = CarbonFootprintTracker()
    
    def add_ev_charging_station(self, spot_id, charging_type, power_level):
        spot = self.spots[spot_id]
        spot.ev_charging = {
            'type': charging_type,
            'power_level': power_level,
            'status': 'available'
        }
        self.ev_charging_spots.append(spot)
    
    def calculate_environmental_impact(self, ticket):
        # Calculate carbon footprint saved vs other transport
        # Track energy usage for EV charging
        # Report sustainability metrics to users
        pass
```

---

## 📊 Performance Benchmarks

### **Expected Performance Metrics:**

| Operation | Target Time | Scalability |
|-----------|-------------|-------------|
| Vehicle Entry | < 2 seconds | 1000 concurrent |
| Spot Allocation | < 500ms | 10,000 spots |
| Payment Processing | < 3 seconds | 500 concurrent |
| Availability Query | < 100ms | Real-time updates |
| Report Generation | < 10 seconds | 1M transactions |

### **Memory Usage:**
- Base system: ~50MB for 1000 spots
- Per active ticket: ~2KB
- Analytics data: ~10MB per month
- Caching overhead: ~20% of base memory

---

## 🎓 Mastery Checklist

### **Core Concepts Covered:**
- ✅ Object-oriented design with inheritance and composition
- ✅ Strategy pattern for algorithms and pricing
- ✅ State management and transitions
- ✅ Thread-safe concurrent operations
- ✅ Payment processing workflows
- ✅ Real-time availability tracking
- ✅ Multi-level architecture design

### **Advanced Features Demonstrated:**
- ✅ Dynamic pricing based on demand
- ✅ License plate recognition integration
- ✅ Mobile API design
- ✅ Multi-tenant support
- ✅ Analytics and business intelligence
- ✅ Security and access control
- ✅ Failure handling and recovery

### **Industry Best Practices:**
- ✅ Clean code organization
- ✅ Comprehensive error handling
- ✅ Performance optimization
- ✅ Scalability considerations
- ✅ Security implementation
- ✅ Testing strategies
- ✅ Documentation and maintainability

---

## 🚀 Next Steps for Production

1. **Database Integration**: Replace in-memory storage with PostgreSQL/MySQL
2. **Message Queue**: Add Redis/RabbitMQ for async operations
3. **Monitoring**: Implement Prometheus/Grafana for system monitoring
4. **Load Balancing**: Add HAProxy/Nginx for high availability
5. **Containerization**: Docker containers with Kubernetes orchestration
6. **CI/CD Pipeline**: Automated testing and deployment
7. **Security Hardening**: OAuth2, rate limiting, input validation
8. **Performance Tuning**: Database indexing, query optimization, caching

This comprehensive parking lot system demonstrates production-ready code with enterprise-level considerations, making it an excellent foundation for system design interviews and real-world implementations.