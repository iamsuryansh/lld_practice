"""
Vending Machine System - Single File Implementation
For coding interviews and production-ready reference

Features:
- State machine for transaction lifecycle
- Multiple payment methods (cash, card, digital)
- Inventory management with thread safety
- Change calculation with greedy algorithm
- Error handling and refund logic

Interview Focus:
- State pattern implementation
- Strategy pattern for payments
- Concurrency and atomicity
- Change-making algorithms
- Error recovery mechanisms

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from threading import RLock
import time
import uuid


# ============================================================================
# MODELS - Core data classes and enums
# ============================================================================

class PaymentType(Enum):
    """Supported payment methods"""
    CASH = "cash"
    CARD = "card"
    DIGITAL = "digital"


class TransactionStatus(Enum):
    """Transaction lifecycle states"""
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Product:
    """
    Product in vending machine inventory
    
    Interview Focus: How do you model inventory items?
    """
    product_id: str
    name: str
    price: float
    category: str
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive")


@dataclass
class InventorySlot:
    """
    Physical slot in vending machine
    
    Interview Focus: How do you handle physical constraints?
    
    Key Insight: Slots have capacity limits and can become inoperable
    """
    slot_id: str
    product: Optional[Product]
    quantity: int
    max_capacity: int
    is_operational: bool = True
    
    def can_dispense(self) -> bool:
        """Check if slot can dispense product"""
        return (self.is_operational and 
                self.quantity > 0 and 
                self.product is not None)
    
    def dispense(self) -> Optional[Product]:
        """
        Attempt to dispense product
        
        Returns:
            Product if successful, None if failed
        """
        if not self.can_dispense():
            return None
        
        self.quantity -= 1
        return self.product


@dataclass
class Transaction:
    """
    Complete transaction record for audit trail
    
    Interview Focus: What information do you track for transactions?
    """
    transaction_id: str
    product_id: str
    slot_id: str
    amount_paid: float
    product_price: float
    payment_type: PaymentType
    status: TransactionStatus
    timestamp: float = field(default_factory=time.time)
    change_given: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def change_due(self) -> float:
        """Calculate change owed to customer"""
        return max(0, self.amount_paid - self.product_price)


# ============================================================================
# PAYMENT - Payment processing strategies
# ============================================================================

class PaymentProcessor(ABC):
    """
    Abstract payment processor
    
    Strategy Pattern: Different payment methods have different processing logic
    
    Interview Focus: How do you handle multiple payment types?
    """
    
    @abstractmethod
    def validate_payment(self, amount: float) -> Tuple[bool, str]:
        """Validate payment before processing"""
        pass
    
    @abstractmethod
    def process_payment(self, amount: float) -> Tuple[bool, str]:
        """Process the payment"""
        pass
    
    @abstractmethod
    def refund(self, amount: float) -> Tuple[bool, str]:
        """Process refund"""
        pass


class CashProcessor(PaymentProcessor):
    """
    Cash payment processor with change calculation
    
    Key Features:
    - Maintains denomination inventory
    - Greedy algorithm for change making
    - Handles insufficient change scenario
    
    Interview Focus: How do you calculate optimal change?
    """
    
    # Standard US currency denominations (in cents to avoid floating point issues)
    DENOMINATIONS = [2000, 1000, 500, 100, 25, 10, 5, 1]  # $20, $10, $5, $1, quarters, dimes, nickels, pennies
    DENOMINATION_NAMES = {
        2000: "$20 bill",
        1000: "$10 bill",
        500: "$5 bill",
        100: "$1 bill",
        25: "quarter",
        10: "dime",
        5: "nickel",
        1: "penny"
    }
    
    def __init__(self):
        # Initialize cash inventory (denomination -> count)
        self.cash_inventory = {
            2000: 10,  # $20 bills
            1000: 20,  # $10 bills
            500: 20,   # $5 bills
            100: 50,   # $1 bills
            25: 100,   # Quarters
            10: 100,   # Dimes
            5: 100,    # Nickels
            1: 200     # Pennies
        }
        self.lock = RLock()
    
    def validate_payment(self, amount: float) -> Tuple[bool, str]:
        """Validate cash payment"""
        if amount <= 0:
            return False, "Invalid payment amount"
        return True, "Payment valid"
    
    def process_payment(self, amount: float) -> Tuple[bool, str]:
        """
        Process cash payment
        
        In real system, this would interface with bill validator / coin acceptor
        """
        valid, message = self.validate_payment(amount)
        if not valid:
            return False, message
        
        # Simulate adding cash to machine inventory
        # In reality, we'd know exact denominations inserted
        return True, f"Cash payment of ${amount:.2f} accepted"
    
    def calculate_change(self, change_amount: float) -> Tuple[bool, Dict[int, int], str]:
        """
        Calculate change using greedy algorithm
        
        Args:
            change_amount: Amount of change to return (in dollars)
            
        Returns:
            (success, change_breakdown, message)
            
        Interview Focus: Explain greedy algorithm for change-making
        
        Time Complexity: O(n) where n is number of denominations
        Space Complexity: O(n) for change breakdown
        
        Key Insight: Greedy works for canonical coin systems (like US currency)
        but not for all denomination sets. For non-canonical, need dynamic programming.
        """
        if change_amount < 0.01:  # Less than 1 cent
            return True, {}, "No change needed"
        
        with self.lock:
            # Convert to cents to avoid floating point errors
            change_cents = int(round(change_amount * 100))
            change_breakdown = {}
            remaining = change_cents
            
            # Greedy algorithm: use largest denominations first
            for denom in self.DENOMINATIONS:
                if remaining == 0:
                    break
                
                # How many of this denomination can we use?
                available = self.cash_inventory[denom]
                needed = remaining // denom
                to_give = min(needed, available)
                
                if to_give > 0:
                    change_breakdown[denom] = to_give
                    remaining -= to_give * denom
                    self.cash_inventory[denom] -= to_give
            
            # Check if we made exact change
            if remaining > 0:
                # Rollback - add denominations back
                for denom, count in change_breakdown.items():
                    self.cash_inventory[denom] += count
                
                return False, {}, f"Cannot make exact change (short ${remaining/100:.2f})"
            
            # Success
            change_description = ", ".join([
                f"{count}x {self.DENOMINATION_NAMES[denom]}" 
                for denom, count in sorted(change_breakdown.items(), reverse=True)
            ])
            
            return True, change_breakdown, f"Change: {change_description}"
    
    def refund(self, amount: float) -> Tuple[bool, str]:
        """Process cash refund by returning exact amount"""
        success, change_breakdown, message = self.calculate_change(amount)
        if success:
            return True, f"Refund processed: ${amount:.2f}"
        else:
            return False, f"Cannot process refund: {message}"
    
    def get_total_cash(self) -> float:
        """Get total cash value in machine"""
        with self.lock:
            total_cents = sum(denom * count for denom, count in self.cash_inventory.items())
            return total_cents / 100.0


class CardProcessor(PaymentProcessor):
    """
    Credit/Debit card payment processor
    
    Interview Focus: How do you handle external payment gateways?
    """
    
    def __init__(self):
        self.processed_transactions = {}
    
    def validate_payment(self, amount: float) -> Tuple[bool, str]:
        """Validate card payment"""
        if amount <= 0:
            return False, "Invalid payment amount"
        # In real system: check card validity, sufficient funds, etc.
        return True, "Card valid"
    
    def process_payment(self, amount: float) -> Tuple[bool, str]:
        """
        Process card payment
        
        In real system: contact payment gateway, handle authorization
        """
        valid, message = self.validate_payment(amount)
        if not valid:
            return False, message
        
        # Simulate authorization
        auth_code = f"AUTH{uuid.uuid4().hex[:8].upper()}"
        self.processed_transactions[auth_code] = {
            'amount': amount,
            'timestamp': time.time()
        }
        
        return True, f"Card payment processed (Auth: {auth_code})"
    
    def refund(self, amount: float) -> Tuple[bool, str]:
        """Process card refund"""
        # In real system: initiate refund through payment gateway
        refund_id = f"REF{uuid.uuid4().hex[:8].upper()}"
        return True, f"Card refund initiated (Refund ID: {refund_id})"


class DigitalProcessor(PaymentProcessor):
    """
    Digital wallet payment processor (Apple Pay, Google Pay, etc.)
    
    Interview Focus: How do you handle modern payment methods?
    """
    
    def validate_payment(self, amount: float) -> Tuple[bool, str]:
        """Validate digital payment"""
        if amount <= 0:
            return False, "Invalid payment amount"
        return True, "Digital payment valid"
    
    def process_payment(self, amount: float) -> Tuple[bool, str]:
        """Process digital wallet payment"""
        valid, message = self.validate_payment(amount)
        if not valid:
            return False, message
        
        token = f"DIG{uuid.uuid4().hex[:8].upper()}"
        return True, f"Digital payment processed (Token: {token})"
    
    def refund(self, amount: float) -> Tuple[bool, str]:
        """Process digital wallet refund"""
        refund_token = f"DIGREF{uuid.uuid4().hex[:8].upper()}"
        return True, f"Digital refund processed (Token: {refund_token})"


# ============================================================================
# STATE MACHINE - Vending machine states
# ============================================================================

class VendingMachineState(ABC):
    """
    Abstract state for vending machine
    
    State Pattern: Encapsulates state-specific behavior
    
    Interview Focus: How do you model state machine transitions?
    
    Key States:
    - IDLE: Ready for new transaction
    - PAYMENT_PENDING: Waiting for payment
    - PAYMENT_RECEIVED: Payment received, product selection needed
    - DISPENSING: Dispensing product and change
    - ERROR: Something went wrong
    """
    
    @abstractmethod
    def insert_payment(self, machine, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        """Handle payment insertion"""
        pass
    
    @abstractmethod
    def select_product(self, machine, slot_id: str) -> Tuple[bool, str]:
        """Handle product selection"""
        pass
    
    @abstractmethod
    def cancel(self, machine) -> Tuple[bool, str]:
        """Handle cancellation"""
        pass
    
    @abstractmethod
    def get_state_name(self) -> str:
        """Get state name for display"""
        pass


class IdleState(VendingMachineState):
    """Machine is idle and ready for transaction"""
    
    def insert_payment(self, machine, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        """Accept payment and transition to payment received state"""
        machine.current_payment_amount = amount
        machine.current_payment_type = payment_type
        machine.set_state(machine.payment_received_state)
        return True, f"Payment of ${amount:.2f} received. Please select product."
    
    def select_product(self, machine, slot_id: str) -> Tuple[bool, str]:
        """Cannot select product without payment"""
        return False, "Please insert payment first"
    
    def cancel(self, machine) -> Tuple[bool, str]:
        """Nothing to cancel"""
        return True, "Machine is idle"
    
    def get_state_name(self) -> str:
        return "IDLE"


class PaymentReceivedState(VendingMachineState):
    """Payment received, waiting for product selection"""
    
    def insert_payment(self, machine, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        """Additional payment not allowed in this implementation"""
        return False, "Payment already received. Please select product or cancel."
    
    def select_product(self, machine, slot_id: str) -> Tuple[bool, str]:
        """Process product selection and dispense"""
        return machine._dispense_product(slot_id)
    
    def cancel(self, machine) -> Tuple[bool, str]:
        """Cancel and refund payment"""
        success, message = machine._refund_payment()
        machine.set_state(machine.idle_state)
        return success, message
    
    def get_state_name(self) -> str:
        return "PAYMENT_RECEIVED"


class DispensingState(VendingMachineState):
    """Machine is dispensing product"""
    
    def insert_payment(self, machine, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        return False, "Transaction in progress"
    
    def select_product(self, machine, slot_id: str) -> Tuple[bool, str]:
        return False, "Transaction in progress"
    
    def cancel(self, machine) -> Tuple[bool, str]:
        return False, "Cannot cancel while dispensing"
    
    def get_state_name(self) -> str:
        return "DISPENSING"


class ErrorState(VendingMachineState):
    """Machine encountered an error"""
    
    def insert_payment(self, machine, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        return False, "Machine out of order"
    
    def select_product(self, machine, slot_id: str) -> Tuple[bool, str]:
        return False, "Machine out of order"
    
    def cancel(self, machine) -> Tuple[bool, str]:
        # Try to refund if payment exists
        if machine.current_payment_amount > 0:
            machine._refund_payment()
        machine.set_state(machine.idle_state)
        return True, "Transaction cancelled"
    
    def get_state_name(self) -> str:
        return "ERROR"


# ============================================================================
# VENDING MACHINE - Main controller
# ============================================================================

class VendingMachine:
    """
    Main vending machine controller
    
    Responsibilities:
    - Manage inventory
    - Process payments
    - Dispense products
    - Handle state transitions
    - Maintain transaction records
    
    Thread Safety: Uses RLock for all operations
    
    Interview Focus: How do you coordinate state machine + payments + inventory?
    """
    
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        
        # Inventory
        self.inventory: Dict[str, InventorySlot] = {}
        
        # Payment processors (Strategy Pattern)
        self.payment_processors = {
            PaymentType.CASH: CashProcessor(),
            PaymentType.CARD: CardProcessor(),
            PaymentType.DIGITAL: DigitalProcessor()
        }
        
        # Current transaction state
        self.current_payment_amount = 0.0
        self.current_payment_type: Optional[PaymentType] = None
        self.selected_slot: Optional[str] = None
        
        # State machine (State Pattern)
        self.idle_state = IdleState()
        self.payment_received_state = PaymentReceivedState()
        self.dispensing_state = DispensingState()
        self.error_state = ErrorState()
        
        self.current_state = self.idle_state
        
        # Transaction history (audit trail)
        self.transactions: List[Transaction] = []
        
        # Thread safety
        self.lock = RLock()
    
    def add_inventory_slot(self, slot: InventorySlot):
        """
        Add inventory slot to machine
        
        Interview Focus: How do you initialize inventory?
        """
        with self.lock:
            self.inventory[slot.slot_id] = slot
    
    def restock_slot(self, slot_id: str, quantity: int) -> Tuple[bool, str]:
        """
        Restock a specific slot
        
        Interview Focus: How do you handle restocking operations?
        """
        with self.lock:
            if slot_id not in self.inventory:
                return False, f"Slot {slot_id} not found"
            
            slot = self.inventory[slot_id]
            new_quantity = min(slot.quantity + quantity, slot.max_capacity)
            actual_added = new_quantity - slot.quantity
            slot.quantity = new_quantity
            
            return True, f"Added {actual_added} items to slot {slot_id}"
    
    def set_state(self, new_state: VendingMachineState):
        """Change machine state"""
        self.current_state = new_state
    
    def insert_payment(self, amount: float, payment_type: PaymentType) -> Tuple[bool, str]:
        """
        Insert payment into machine
        
        Interview Focus: How do you handle payment in state machine?
        """
        with self.lock:
            # Validate payment amount
            if amount <= 0:
                return False, "Invalid payment amount"
            
            # Delegate to current state
            return self.current_state.insert_payment(self, amount, payment_type)
    
    def select_product(self, slot_id: str) -> Tuple[bool, str]:
        """
        Select product for purchase
        
        Interview Focus: How do you handle product selection?
        """
        with self.lock:
            # Delegate to current state
            return self.current_state.select_product(self, slot_id)
    
    def cancel_transaction(self) -> Tuple[bool, str]:
        """
        Cancel current transaction
        
        Interview Focus: How do you handle cancellations?
        """
        with self.lock:
            return self.current_state.cancel(self)
    
    def _dispense_product(self, slot_id: str) -> Tuple[bool, str]:
        """
        Internal method to dispense product
        
        Interview Focus: How do you handle the critical dispense operation?
        
        Key Challenges:
        - Ensure payment is sufficient
        - Calculate change
        - Handle dispensing failure
        - Update inventory atomically
        """
        # Validate slot
        if slot_id not in self.inventory:
            self._handle_error(f"Slot {slot_id} not found")
            return False, f"Invalid slot {slot_id}"
        
        slot = self.inventory[slot_id]
        
        # Check if slot can dispense
        if not slot.can_dispense():
            self._handle_error(f"Slot {slot_id} cannot dispense")
            return False, f"Product unavailable in slot {slot_id}"
        
        product = slot.product
        
        # Check if payment is sufficient
        if self.current_payment_amount < product.price:
            self._handle_error("Insufficient payment")
            return False, f"Insufficient payment. Need ${product.price:.2f}, have ${self.current_payment_amount:.2f}"
        
        # Change to dispensing state
        self.set_state(self.dispensing_state)
        
        # Calculate change
        change_due = self.current_payment_amount - product.price
        
        # Process payment through processor
        processor = self.payment_processors[self.current_payment_type]
        payment_success, payment_msg = processor.process_payment(self.current_payment_amount)
        
        if not payment_success:
            self._handle_error(f"Payment processing failed: {payment_msg}")
            return False, f"Payment failed: {payment_msg}"
        
        # Calculate change if needed
        change_breakdown = {}
        if change_due > 0.01 and self.current_payment_type == PaymentType.CASH:
            cash_processor = processor
            success, change_breakdown, change_msg = cash_processor.calculate_change(change_due)
            
            if not success:
                # Cannot make change - refund entire amount
                self._refund_payment()
                self._handle_error(f"Cannot make change: {change_msg}")
                return False, f"Cannot make exact change. Transaction cancelled."
        
        # Dispense product
        dispensed_product = slot.dispense()
        
        if not dispensed_product:
            # Dispensing failed - refund
            self._refund_payment()
            self._handle_error("Mechanical dispensing failure")
            return False, "Dispensing failed. Payment refunded."
        
        # Create transaction record
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            product_id=product.product_id,
            slot_id=slot_id,
            amount_paid=self.current_payment_amount,
            product_price=product.price,
            payment_type=self.current_payment_type,
            status=TransactionStatus.COMPLETED,
            change_given=change_due
        )
        self.transactions.append(transaction)
        
        # Build success message
        message = f"✅ Dispensed: {product.name}"
        if change_due > 0.01:
            if change_breakdown:
                change_desc = ", ".join([
                    f"{count}x {CashProcessor.DENOMINATION_NAMES[denom]}"
                    for denom, count in sorted(change_breakdown.items(), reverse=True)
                ])
                message += f"\n💰 Change: ${change_due:.2f} ({change_desc})"
            else:
                message += f"\n💰 Change: ${change_due:.2f}"
        
        # Reset state
        self._reset_transaction()
        
        return True, message
    
    def _refund_payment(self) -> Tuple[bool, str]:
        """
        Refund current payment
        
        Interview Focus: How do you handle refunds?
        """
        if self.current_payment_amount <= 0:
            return True, "No payment to refund"
        
        processor = self.payment_processors[self.current_payment_type]
        success, message = processor.refund(self.current_payment_amount)
        
        if success:
            # Create refund transaction record
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                product_id="",
                slot_id="",
                amount_paid=self.current_payment_amount,
                product_price=0,
                payment_type=self.current_payment_type,
                status=TransactionStatus.REFUNDED
            )
            self.transactions.append(transaction)
        
        return success, message
    
    def _handle_error(self, error_message: str):
        """Handle error condition"""
        print(f"❌ ERROR: {error_message}")
        
        # Try to refund payment
        if self.current_payment_amount > 0:
            self._refund_payment()
        
        self.set_state(self.error_state)
    
    def _reset_transaction(self):
        """Reset transaction state"""
        self.current_payment_amount = 0.0
        self.current_payment_type = None
        self.selected_slot = None
        self.set_state(self.idle_state)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get machine status
        
        Interview Focus: What information do you expose for monitoring?
        """
        with self.lock:
            available_products = [
                {
                    'slot': slot_id,
                    'product': slot.product.name,
                    'price': slot.product.price,
                    'quantity': slot.quantity
                }
                for slot_id, slot in self.inventory.items()
                if slot.can_dispense()
            ]
            
            cash_processor = self.payment_processors[PaymentType.CASH]
            
            return {
                'machine_id': self.machine_id,
                'state': self.current_state.get_state_name(),
                'available_products': available_products,
                'total_slots': len(self.inventory),
                'cash_balance': cash_processor.get_total_cash(),
                'total_transactions': len(self.transactions),
                'current_payment': self.current_payment_amount
            }
    
    def get_sales_summary(self) -> Dict[str, Any]:
        """Get sales summary for reporting"""
        with self.lock:
            completed = [t for t in self.transactions if t.status == TransactionStatus.COMPLETED]
            
            total_revenue = sum(t.product_price for t in completed)
            total_transactions = len(completed)
            
            # Group by product
            product_sales = defaultdict(int)
            for t in completed:
                product_sales[t.product_id] += 1
            
            # Group by payment type
            payment_breakdown = defaultdict(int)
            for t in completed:
                payment_breakdown[t.payment_type.value] += 1
            
            return {
                'total_revenue': total_revenue,
                'total_transactions': total_transactions,
                'avg_transaction_value': total_revenue / total_transactions if total_transactions > 0 else 0,
                'product_sales': dict(product_sales),
                'payment_methods': dict(payment_breakdown)
            }


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
    """Demonstrate basic vending machine operations"""
    print_separator("Basic Vending Machine Operations")
    
    # Initialize machine
    machine = VendingMachine("VM001")
    
    # Add inventory
    coke = Product("P001", "Coca Cola", 1.50, "Beverages")
    chips = Product("P002", "Potato Chips", 2.00, "Snacks")
    water = Product("P003", "Water", 1.00, "Beverages")
    
    machine.add_inventory_slot(InventorySlot("A1", coke, 10, 15))
    machine.add_inventory_slot(InventorySlot("B1", chips, 8, 12))
    machine.add_inventory_slot(InventorySlot("C1", water, 15, 20))
    
    print("\n📦 Inventory loaded:")
    status = machine.get_status()
    for product in status['available_products']:
        print(f"  {product['slot']}: {product['product']} - ${product['price']:.2f} (Qty: {product['quantity']})")
    
    # Test purchase with exact change
    print("\n💵 Test 1: Purchase with exact cash")
    print("  Inserting $1.50 for Coca Cola...")
    success, msg = machine.insert_payment(1.50, PaymentType.CASH)
    print(f"  {msg}")
    
    print("  Selecting product A1...")
    success, msg = machine.select_product("A1")
    print(f"  {msg}")
    
    # Test purchase with change needed
    print("\n💵 Test 2: Purchase with change")
    print("  Inserting $5.00 for Potato Chips ($2.00)...")
    success, msg = machine.insert_payment(5.00, PaymentType.CASH)
    print(f"  {msg}")
    
    print("  Selecting product B1...")
    success, msg = machine.select_product("B1")
    print(f"  {msg}")


def demo_state_machine():
    """Demonstrate state machine behavior"""
    print_separator("State Machine Demonstration")
    
    machine = VendingMachine("VM002")
    
    # Add inventory
    water = Product("P001", "Water", 1.00, "Beverages")
    machine.add_inventory_slot(InventorySlot("A1", water, 5, 10))
    
    print("\n🔄 State Transition Test:")
    print(f"  Current state: {machine.current_state.get_state_name()}")
    
    print("\n  1. Try to select product without payment:")
    success, msg = machine.select_product("A1")
    print(f"     Result: {msg}")
    print(f"     State: {machine.current_state.get_state_name()}")
    
    print("\n  2. Insert payment:")
    success, msg = machine.insert_payment(2.00, PaymentType.CASH)
    print(f"     Result: {msg}")
    print(f"     State: {machine.current_state.get_state_name()}")
    
    print("\n  3. Cancel transaction:")
    success, msg = machine.cancel_transaction()
    print(f"     Result: {msg}")
    print(f"     State: {machine.current_state.get_state_name()}")


def demo_error_handling():
    """Demonstrate error handling"""
    print_separator("Error Handling Demonstration")
    
    machine = VendingMachine("VM003")
    
    # Add inventory
    chips = Product("P001", "Chips", 2.50, "Snacks")
    machine.add_inventory_slot(InventorySlot("A1", chips, 1, 10))
    
    print("\n⚠️ Error Scenario 1: Insufficient payment")
    machine.insert_payment(2.00, PaymentType.CASH)
    success, msg = machine.select_product("A1")
    print(f"  {msg}")
    print(f"  State after error: {machine.current_state.get_state_name()}")
    
    print("\n⚠️ Error Scenario 2: Invalid slot")
    machine._reset_transaction()
    machine.insert_payment(3.00, PaymentType.CASH)
    success, msg = machine.select_product("Z9")
    print(f"  {msg}")
    print(f"  State after error: {machine.current_state.get_state_name()}")


def demo_payment_methods():
    """Demonstrate different payment methods"""
    print_separator("Payment Methods Demonstration")
    
    machine = VendingMachine("VM004")
    
    # Add inventory
    product = Product("P001", "Snack", 1.50, "Snacks")
    machine.add_inventory_slot(InventorySlot("A1", product, 10, 15))
    
    print("\n💳 Test 1: Card payment")
    machine.insert_payment(1.50, PaymentType.CARD)
    success, msg = machine.select_product("A1")
    print(f"  {msg}")
    
    print("\n📱 Test 2: Digital wallet payment")
    machine.insert_payment(1.50, PaymentType.DIGITAL)
    success, msg = machine.select_product("A1")
    print(f"  {msg}")
    
    print("\n💵 Test 3: Cash payment with change")
    machine.insert_payment(2.00, PaymentType.CASH)
    success, msg = machine.select_product("A1")
    print(f"  {msg}")


def demo_sales_reporting():
    """Demonstrate sales reporting"""
    print_separator("Sales Reporting")
    
    machine = VendingMachine("VM005")
    
    # Add inventory
    coke = Product("P001", "Coca Cola", 1.50, "Beverages")
    chips = Product("P002", "Chips", 2.00, "Snacks")
    
    machine.add_inventory_slot(InventorySlot("A1", coke, 5, 10))
    machine.add_inventory_slot(InventorySlot("B1", chips, 5, 10))
    
    # Simulate some sales
    print("\n💰 Simulating sales...")
    
    machine.insert_payment(1.50, PaymentType.CASH)
    machine.select_product("A1")
    
    machine.insert_payment(2.00, PaymentType.CARD)
    machine.select_product("B1")
    
    machine.insert_payment(1.50, PaymentType.DIGITAL)
    machine.select_product("A1")
    
    # Print summary
    summary = machine.get_sales_summary()
    print(f"\n📊 Sales Summary:")
    print(f"  Total Revenue: ${summary['total_revenue']:.2f}")
    print(f"  Total Transactions: {summary['total_transactions']}")
    print(f"  Average Transaction: ${summary['avg_transaction_value']:.2f}")
    print(f"\n  Product Sales:")
    for product_id, count in summary['product_sales'].items():
        print(f"    {product_id}: {count} sold")
    print(f"\n  Payment Methods:")
    for method, count in summary['payment_methods'].items():
        print(f"    {method}: {count} transactions")


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  VENDING MACHINE SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("  Features: State machine, Multiple payments, Change calculation")
    print("="*70)
    
    demo_basic_operations()
    demo_state_machine()
    demo_payment_methods()
    demo_error_handling()
    demo_sales_reporting()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python vending_machine_merged.py
    """
    run_demo()
