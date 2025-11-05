"""
Parking Lot System - Low Level Design Implementation
===================================================

A comprehensive parking lot management system supporting:
- Multi-level parking structure
- Different vehicle types (Car, Bike, Truck)  
- Multiple spot allocation strategies
- Payment processing with different methods
- Real-time availability tracking
- Thread-safe concurrent operations

Interview Focus Areas:
- Object-oriented design principles
- Strategy pattern for allocation algorithms
- State management and transitions
- Concurrency and thread safety
- Payment processing workflows
- System scalability considerations

Author: System Design Interview Practice
Date: November 2025
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import threading
import time
import uuid
from collections import defaultdict
import heapq


class VehicleType(Enum):
    """Enumeration for different vehicle types"""
    BIKE = "BIKE"
    CAR = "CAR" 
    TRUCK = "TRUCK"


class SpotType(Enum):
    """Enumeration for different parking spot types"""
    BIKE_SPOT = "BIKE_SPOT"
    COMPACT_SPOT = "COMPACT_SPOT"
    REGULAR_SPOT = "REGULAR_SPOT"
    LARGE_SPOT = "LARGE_SPOT"


class PaymentMethod(Enum):
    """Enumeration for payment methods"""
    CASH = "CASH"
    CREDIT_CARD = "CREDIT_CARD"
    MOBILE_PAYMENT = "MOBILE_PAYMENT"


class ParkingSpotStatus(Enum):
    """Enumeration for parking spot status"""
    AVAILABLE = "AVAILABLE"
    OCCUPIED = "OCCUPIED"
    RESERVED = "RESERVED"
    OUT_OF_ORDER = "OUT_OF_ORDER"


class PaymentStatus(Enum):
    """Enumeration for payment status"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"


@dataclass
class Vehicle:
    """Represents a vehicle in the parking system"""
    license_plate: str
    vehicle_type: VehicleType
    owner_name: str = ""
    phone_number: str = ""
    
    def __hash__(self):
        return hash(self.license_plate)
    
    def __eq__(self, other):
        return isinstance(other, Vehicle) and self.license_plate == other.license_plate


@dataclass
class ParkingSpot:
    """Represents a parking spot"""
    spot_id: str
    spot_type: SpotType
    floor_number: int
    section: str
    status: ParkingSpotStatus = ParkingSpotStatus.AVAILABLE
    vehicle: Optional[Vehicle] = None
    reserved_until: Optional[datetime] = None
    
    def is_available(self) -> bool:
        """Check if spot is available for parking"""
        if self.status == ParkingSpotStatus.AVAILABLE:
            return True
        if self.status == ParkingSpotStatus.RESERVED and self.reserved_until:
            return datetime.now() > self.reserved_until
        return False
    
    def can_fit_vehicle(self, vehicle_type: VehicleType) -> bool:
        """Check if spot can accommodate the vehicle type"""
        compatibility = {
            SpotType.BIKE_SPOT: [VehicleType.BIKE],
            SpotType.COMPACT_SPOT: [VehicleType.BIKE, VehicleType.CAR],
            SpotType.REGULAR_SPOT: [VehicleType.BIKE, VehicleType.CAR],
            SpotType.LARGE_SPOT: [VehicleType.BIKE, VehicleType.CAR, VehicleType.TRUCK]
        }
        return vehicle_type in compatibility.get(self.spot_type, [])


@dataclass
class Payment:
    """Represents a payment transaction"""
    payment_id: str
    amount: float
    method: PaymentMethod
    status: PaymentStatus = PaymentStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    transaction_details: Dict = field(default_factory=dict)


@dataclass
class ParkingTicket:
    """Represents a parking ticket"""
    ticket_id: str
    vehicle: Vehicle
    spot: ParkingSpot
    entry_time: datetime
    exit_time: Optional[datetime] = None
    payment: Optional[Payment] = None
    total_cost: float = 0.0
    
    def calculate_parking_duration(self) -> timedelta:
        """Calculate parking duration"""
        end_time = self.exit_time or datetime.now()
        return end_time - self.entry_time
    
    def is_paid(self) -> bool:
        """Check if ticket is paid"""
        return self.payment and self.payment.status == PaymentStatus.COMPLETED


class PricingStrategy(ABC):
    """Abstract base class for pricing strategies"""
    
    @abstractmethod
    def calculate_cost(self, vehicle_type: VehicleType, duration: timedelta) -> float:
        """Calculate parking cost based on vehicle type and duration"""
        pass


class HourlyPricingStrategy(PricingStrategy):
    """Hourly pricing strategy"""
    
    def __init__(self):
        self.hourly_rates = {
            VehicleType.BIKE: 2.0,
            VehicleType.CAR: 5.0,
            VehicleType.TRUCK: 10.0
        }
    
    def calculate_cost(self, vehicle_type: VehicleType, duration: timedelta) -> float:
        """Calculate cost based on hourly rates"""
        hours = max(1, int(duration.total_seconds() / 3600))  # Minimum 1 hour
        return hours * self.hourly_rates[vehicle_type]


class FlatRatePricingStrategy(PricingStrategy):
    """Flat rate pricing strategy"""
    
    def __init__(self):
        self.flat_rates = {
            VehicleType.BIKE: 10.0,
            VehicleType.CAR: 25.0,
            VehicleType.TRUCK: 50.0
        }
    
    def calculate_cost(self, vehicle_type: VehicleType, duration: timedelta) -> float:
        """Calculate flat rate cost"""
        return self.flat_rates[vehicle_type]


class SpotAllocationStrategy(ABC):
    """Abstract base class for spot allocation strategies"""
    
    @abstractmethod
    def find_spot(self, vehicle_type: VehicleType, available_spots: List[ParkingSpot]) -> Optional[ParkingSpot]:
        """Find the best spot for the vehicle"""
        pass


class ClosestToEntranceStrategy(SpotAllocationStrategy):
    """Allocate spots closest to entrance (lower floor numbers first)"""
    
    def find_spot(self, vehicle_type: VehicleType, available_spots: List[ParkingSpot]) -> Optional[ParkingSpot]:
        """Find closest spot to entrance"""
        suitable_spots = [spot for spot in available_spots if spot.can_fit_vehicle(vehicle_type)]
        if not suitable_spots:
            return None
        
        # Sort by floor number (ascending) then by spot_id
        suitable_spots.sort(key=lambda x: (x.floor_number, x.spot_id))
        return suitable_spots[0]


class OptimalFitStrategy(SpotAllocationStrategy):
    """Allocate smallest suitable spot to maximize space utilization"""
    
    def find_spot(self, vehicle_type: VehicleType, available_spots: List[ParkingSpot]) -> Optional[ParkingSpot]:
        """Find optimal fit spot"""
        suitable_spots = [spot for spot in available_spots if spot.can_fit_vehicle(vehicle_type)]
        if not suitable_spots:
            return None
        
        # Define spot size priority (smaller spots first)
        spot_priority = {
            SpotType.BIKE_SPOT: 1,
            SpotType.COMPACT_SPOT: 2, 
            SpotType.REGULAR_SPOT: 3,
            SpotType.LARGE_SPOT: 4
        }
        
        # Sort by spot size then by floor number
        suitable_spots.sort(key=lambda x: (spot_priority[x.spot_type], x.floor_number, x.spot_id))
        return suitable_spots[0]


class PaymentProcessor(ABC):
    """Abstract base class for payment processing"""
    
    @abstractmethod
    def process_payment(self, amount: float, method: PaymentMethod, details: Dict) -> Payment:
        """Process payment and return payment object"""
        pass


class SimplePaymentProcessor(PaymentProcessor):
    """Simple payment processor for demonstration"""
    
    def process_payment(self, amount: float, method: PaymentMethod, details: Dict) -> Payment:
        """Process payment with simulated success/failure"""
        payment = Payment(
            payment_id=str(uuid.uuid4()),
            amount=amount,
            method=method,
            transaction_details=details
        )
        
        # Simulate payment processing (90% success rate)
        import random
        if random.random() < 0.9:
            payment.status = PaymentStatus.COMPLETED
        else:
            payment.status = PaymentStatus.FAILED
        
        return payment


class ParkingLotSystem:
    """Main parking lot system managing all operations"""
    
    def __init__(self, name: str, total_floors: int):
        self.name = name
        self.total_floors = total_floors
        self.spots: Dict[str, ParkingSpot] = {}
        self.active_tickets: Dict[str, ParkingTicket] = {}
        self.completed_tickets: Dict[str, ParkingTicket] = {}
        self.vehicle_to_ticket: Dict[Vehicle, ParkingTicket] = {}
        
        # Strategies (configurable)
        self.allocation_strategy: SpotAllocationStrategy = OptimalFitStrategy()
        self.pricing_strategy: PricingStrategy = HourlyPricingStrategy()
        self.payment_processor: PaymentProcessor = SimplePaymentProcessor()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Initialize with default spots
        self._initialize_default_spots()
    
    def _initialize_default_spots(self):
        """Initialize parking lot with default spot configuration"""
        spot_configs = [
            (SpotType.BIKE_SPOT, 20),      # 20 bike spots per floor
            (SpotType.COMPACT_SPOT, 30),   # 30 compact spots per floor
            (SpotType.REGULAR_SPOT, 40),   # 40 regular spots per floor
            (SpotType.LARGE_SPOT, 10)      # 10 large spots per floor
        ]
        
        for floor in range(1, self.total_floors + 1):
            for spot_type, count in spot_configs:
                section = spot_type.value[0]  # First letter as section
                for i in range(1, count + 1):
                    spot_id = f"F{floor}{section}{i:03d}"
                    self.add_parking_spot(spot_id, spot_type, floor, section)
    
    def add_parking_spot(self, spot_id: str, spot_type: SpotType, floor_number: int, section: str):
        """Add a parking spot to the system"""
        with self.lock:
            if spot_id not in self.spots:
                self.spots[spot_id] = ParkingSpot(spot_id, spot_type, floor_number, section)
                return True
            return False
    
    def get_available_spots(self) -> List[ParkingSpot]:
        """Get all available parking spots"""
        with self.lock:
            return [spot for spot in self.spots.values() if spot.is_available()]
    
    def get_availability_by_type(self) -> Dict[VehicleType, int]:
        """Get availability count by vehicle type"""
        with self.lock:
            availability = {vtype: 0 for vtype in VehicleType}
            
            for spot in self.get_available_spots():
                for vehicle_type in VehicleType:
                    if spot.can_fit_vehicle(vehicle_type):
                        availability[vehicle_type] += 1
                        break  # Count spot only once for smallest compatible vehicle
            
            return availability
    
    def park_vehicle(self, vehicle: Vehicle) -> Optional[ParkingTicket]:
        """Park a vehicle and return parking ticket"""
        with self.lock:
            # Check if vehicle is already parked
            if vehicle in self.vehicle_to_ticket:
                return None
            
            # Find available spot
            available_spots = self.get_available_spots()
            spot = self.allocation_strategy.find_spot(vehicle.vehicle_type, available_spots)
            
            if not spot:
                return None
            
            # Create parking ticket
            ticket = ParkingTicket(
                ticket_id=str(uuid.uuid4()),
                vehicle=vehicle,
                spot=spot,
                entry_time=datetime.now()
            )
            
            # Update spot and system state
            spot.status = ParkingSpotStatus.OCCUPIED
            spot.vehicle = vehicle
            self.active_tickets[ticket.ticket_id] = ticket
            self.vehicle_to_ticket[vehicle] = ticket
            
            # Update statistics
            self.stats['total_parkings'] += 1
            self.stats[f'{vehicle.vehicle_type.value}_parkings'] += 1
            
            return ticket
    
    def exit_vehicle(self, ticket_id: str) -> Optional[ParkingTicket]:
        """Process vehicle exit and calculate charges"""
        with self.lock:
            ticket = self.active_tickets.get(ticket_id)
            if not ticket:
                return None
            
            # Update ticket with exit time
            ticket.exit_time = datetime.now()
            
            # Calculate cost
            duration = ticket.calculate_parking_duration()
            ticket.total_cost = self.pricing_strategy.calculate_cost(
                ticket.vehicle.vehicle_type, 
                duration
            )
            
            return ticket
    
    def process_payment(self, ticket_id: str, payment_method: PaymentMethod, 
                       payment_details: Dict = None) -> bool:
        """Process payment for parking ticket"""
        with self.lock:
            ticket = self.active_tickets.get(ticket_id)
            if not ticket or ticket.total_cost == 0:
                return False
            
            # Process payment
            payment_details = payment_details or {}
            payment = self.payment_processor.process_payment(
                ticket.total_cost, 
                payment_method, 
                payment_details
            )
            
            ticket.payment = payment
            
            if payment.status == PaymentStatus.COMPLETED:
                # Complete the parking session
                self._complete_parking(ticket)
                return True
            
            return False
    
    def _complete_parking(self, ticket: ParkingTicket):
        """Complete parking session and free up spot"""
        # Move ticket to completed
        self.completed_tickets[ticket.ticket_id] = ticket
        del self.active_tickets[ticket.ticket_id]
        del self.vehicle_to_ticket[ticket.vehicle]
        
        # Free up spot
        ticket.spot.status = ParkingSpotStatus.AVAILABLE
        ticket.spot.vehicle = None
        
        # Update statistics
        self.stats['completed_parkings'] += 1
        self.stats['total_revenue'] += ticket.total_cost
    
    def reserve_spot(self, vehicle_type: VehicleType, duration_minutes: int = 30) -> Optional[str]:
        """Reserve a parking spot for specified duration"""
        with self.lock:
            available_spots = self.get_available_spots()
            spot = self.allocation_strategy.find_spot(vehicle_type, available_spots)
            
            if not spot:
                return None
            
            # Reserve the spot
            spot.status = ParkingSpotStatus.RESERVED
            spot.reserved_until = datetime.now() + timedelta(minutes=duration_minutes)
            
            return spot.spot_id
    
    def get_parking_history(self, vehicle: Vehicle) -> List[ParkingTicket]:
        """Get parking history for a vehicle"""
        with self.lock:
            history = []
            
            # Check active tickets
            if vehicle in self.vehicle_to_ticket:
                history.append(self.vehicle_to_ticket[vehicle])
            
            # Check completed tickets
            for ticket in self.completed_tickets.values():
                if ticket.vehicle == vehicle:
                    history.append(ticket)
            
            # Sort by entry time
            history.sort(key=lambda x: x.entry_time, reverse=True)
            return history
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        with self.lock:
            total_spots = len(self.spots)
            available_spots = len(self.get_available_spots())
            occupancy_rate = ((total_spots - available_spots) / total_spots) * 100
            
            stats = {
                'total_spots': total_spots,
                'available_spots': available_spots,
                'occupied_spots': total_spots - available_spots,
                'occupancy_rate': round(occupancy_rate, 2),
                'availability_by_type': self.get_availability_by_type(),
                'total_parkings': self.stats['total_parkings'],
                'completed_parkings': self.stats['completed_parkings'],
                'active_parkings': len(self.active_tickets),
                'total_revenue': round(self.stats['total_revenue'], 2),
                'parkings_by_vehicle_type': {
                    vtype.value: self.stats[f'{vtype.value}_parkings'] 
                    for vtype in VehicleType
                }
            }
            
            return stats
    
    def set_allocation_strategy(self, strategy: SpotAllocationStrategy):
        """Set the spot allocation strategy"""
        with self.lock:
            self.allocation_strategy = strategy
    
    def set_pricing_strategy(self, strategy: PricingStrategy):
        """Set the pricing strategy"""
        with self.lock:
            self.pricing_strategy = strategy


def demonstrate_parking_lot_system():
    """Demonstration of the parking lot system"""
    print("=== Parking Lot System Demonstration ===\n")
    
    # Create parking lot system
    parking_lot = ParkingLotSystem("City Center Parking", total_floors=3)
    print(f"✅ Created parking lot: {parking_lot.name} with {parking_lot.total_floors} floors")
    
    # Create vehicles
    vehicles = [
        Vehicle("ABC123", VehicleType.CAR, "John Doe", "555-0001"),
        Vehicle("XYZ789", VehicleType.BIKE, "Jane Smith", "555-0002"),
        Vehicle("TRK456", VehicleType.TRUCK, "Bob Johnson", "555-0003"),
        Vehicle("CAR999", VehicleType.CAR, "Alice Wilson", "555-0004")
    ]
    
    print(f"\n📋 Created {len(vehicles)} vehicles for testing")
    
    # Display initial availability
    print("\n🅿️  Initial Parking Availability:")
    availability = parking_lot.get_availability_by_type()
    for vehicle_type, count in availability.items():
        print(f"   {vehicle_type.value}: {count} spots")
    
    # Park vehicles
    print("\n🚗 Parking Vehicles:")
    tickets = []
    for vehicle in vehicles:
        ticket = parking_lot.park_vehicle(vehicle)
        if ticket:
            tickets.append(ticket)
            print(f"   ✅ Parked {vehicle.vehicle_type.value} {vehicle.license_plate} at spot {ticket.spot.spot_id}")
        else:
            print(f"   ❌ Could not park {vehicle.vehicle_type.value} {vehicle.license_plate}")
    
    # Show updated availability
    print("\n📊 Updated Availability:")
    availability = parking_lot.get_availability_by_type()
    for vehicle_type, count in availability.items():
        print(f"   {vehicle_type.value}: {count} spots")
    
    # Simulate some parking time
    print("\n⏰ Simulating parking duration...")
    time.sleep(1)  # Simulate time passing
    
    # Process vehicle exits and payments
    print("\n💳 Processing Exits and Payments:")
    for ticket in tickets[:2]:  # Process first 2 vehicles
        # Calculate exit cost
        exit_ticket = parking_lot.exit_vehicle(ticket.ticket_id)
        if exit_ticket:
            duration = exit_ticket.calculate_parking_duration()
            print(f"   🚪 {exit_ticket.vehicle.license_plate} exiting after {duration}")
            print(f"      💰 Cost: ${exit_ticket.total_cost:.2f}")
            
            # Process payment
            payment_success = parking_lot.process_payment(
                ticket.ticket_id, 
                PaymentMethod.CREDIT_CARD,
                {"card_number": "****1234", "cvv": "***"}
            )
            
            if payment_success:
                print(f"      ✅ Payment processed successfully")
            else:
                print(f"      ❌ Payment failed")
    
    # Test spot reservation
    print("\n🔒 Testing Spot Reservation:")
    reserved_spot = parking_lot.reserve_spot(VehicleType.CAR, duration_minutes=60)
    if reserved_spot:
        print(f"   ✅ Reserved spot {reserved_spot} for 60 minutes")
    else:
        print(f"   ❌ No spots available for reservation")
    
    # Test different allocation strategy
    print("\n🎯 Testing Different Allocation Strategy:")
    parking_lot.set_allocation_strategy(ClosestToEntranceStrategy())
    print("   🔄 Switched to ClosestToEntranceStrategy")
    
    # Try parking another vehicle
    new_vehicle = Vehicle("NEW123", VehicleType.CAR, "New Driver", "555-9999")
    new_ticket = parking_lot.park_vehicle(new_vehicle)
    if new_ticket:
        print(f"   ✅ Parked {new_vehicle.license_plate} at spot {new_ticket.spot.spot_id}")
    
    # Test flat rate pricing
    print("\n💲 Testing Flat Rate Pricing:")
    parking_lot.set_pricing_strategy(FlatRatePricingStrategy())
    print("   🔄 Switched to FlatRatePricingStrategy")
    
    # Display comprehensive statistics
    print("\n📈 Final System Statistics:")
    stats = parking_lot.get_system_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Display parking history for a vehicle
    print(f"\n📜 Parking History for {vehicles[0].license_plate}:")
    history = parking_lot.get_parking_history(vehicles[0])
    for i, ticket in enumerate(history, 1):
        status = "COMPLETED" if ticket.is_paid() else "ACTIVE"
        print(f"   {i}. {ticket.entry_time.strftime('%Y-%m-%d %H:%M')} - Spot {ticket.spot.spot_id} - {status}")
        if ticket.total_cost > 0:
            print(f"      Cost: ${ticket.total_cost:.2f}")


def demonstrate_concurrent_operations():
    """Demonstrate thread-safe concurrent operations"""
    print("\n=== Concurrent Operations Demonstration ===\n")
    
    parking_lot = ParkingLotSystem("Concurrent Test Lot", total_floors=2)
    
    # Create multiple vehicles for concurrent testing
    test_vehicles = [
        Vehicle(f"CONCURRENT{i:03d}", VehicleType.CAR, f"Driver{i}", f"555-{i:04d}")
        for i in range(10)
    ]
    
    print(f"🔄 Testing concurrent parking of {len(test_vehicles)} vehicles...")
    
    successful_parkings = []
    failed_parkings = []
    
    def park_vehicle_thread(vehicle):
        """Thread function for parking a vehicle"""
        ticket = parking_lot.park_vehicle(vehicle)
        if ticket:
            successful_parkings.append(ticket)
        else:
            failed_parkings.append(vehicle)
    
    # Create threads for concurrent parking
    threads = []
    for vehicle in test_vehicles:
        thread = threading.Thread(target=park_vehicle_thread, args=(vehicle,))
        threads.append(thread)
    
    # Start all threads
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    print(f"✅ Concurrent operations completed in {end_time - start_time:.3f} seconds")
    print(f"   Successful parkings: {len(successful_parkings)}")
    print(f"   Failed parkings: {len(failed_parkings)}")
    
    # Verify system consistency
    stats = parking_lot.get_system_statistics()
    print(f"   Active parkings in system: {stats['active_parkings']}")
    print(f"   System consistency: {'✅ PASSED' if len(successful_parkings) == stats['active_parkings'] else '❌ FAILED'}")


if __name__ == "__main__":
    # Run main demonstration
    demonstrate_parking_lot_system()
    
    print("\n" + "="*50)
    
    # Run concurrent operations test
    demonstrate_concurrent_operations()
    
    print("\n🎉 Parking Lot System demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Multi-level parking structure")
    print("✅ Multiple vehicle types support")
    print("✅ Configurable allocation strategies")
    print("✅ Payment processing workflows")
    print("✅ Spot reservation system")
    print("✅ Real-time availability tracking")
    print("✅ Thread-safe concurrent operations")
    print("✅ Comprehensive statistics and reporting")