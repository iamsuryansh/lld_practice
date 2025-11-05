# Vending Machine System - Interview Preparation Guide

**Target Audience**: Software Engineers with 2-5 years of experience  
**Focus**: Object-oriented design, state patterns, payment processing  
**Estimated Study Time**: 4-6 hours

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
3. [Critical Knowledge Points](#critical-knowledge-points)
4. [Expected Interview Questions & Answers](#expected-interview-questions--answers)
5. [Testing Strategy](#testing-strategy)
6. [Production Considerations](#production-considerations)

---

## Problem Statement

Design a vending machine system that can:
- Handle multiple payment methods (cash, card, digital wallet)
- Manage inventory across multiple slots
- Calculate change accurately
- Handle error scenarios (out of stock, insufficient payment, mechanical failures)
- Maintain transaction audit trails
- Support concurrent operations safely

**Core Challenge**: How do you model complex state transitions and coordinate payments, inventory, and dispensing?

---

## Step-by-Step Implementation Guide

### Phase 1: Data Models (15-20 minutes)

**What to do**:
```python
# Define core entities
class Product:
    product_id, name, price, category

class InventorySlot:
    slot_id, product, quantity, max_capacity, is_operational
    
class Transaction:
    transaction_id, product_id, amount_paid, payment_type, status
```

**Why this approach**:
- Separates product (what) from slot (where)
- Allows same product in multiple slots
- Tracks physical constraints (capacity, operational status)
- Transaction records enable audit trail

**Common mistake**: Making Product and InventorySlot the same thing. They serve different purposes.

---

### Phase 2: Payment Strategies (25-30 minutes)

**What to do**:
```python
# Strategy pattern for different payment types
class PaymentProcessor(ABC):
    def validate_payment(self, amount) -> Tuple[bool, str]
    def process_payment(self, amount) -> Tuple[bool, str]
    def refund(self, amount) -> Tuple[bool, str]

class CashProcessor(PaymentProcessor):
    # Key: Change calculation with denomination inventory
    
class CardProcessor(PaymentProcessor):
    # Key: Authorization codes, external gateway integration
    
class DigitalProcessor(PaymentProcessor):
    # Key: Token-based authentication
```

**Why Strategy Pattern**:
- Each payment type has unique processing logic
- Easy to add new payment methods without modifying existing code
- Separates payment logic from vending machine logic

**Interview Insight**: Be ready to explain why inheritance (is-a) vs composition (has-a). Here we use composition: VendingMachine *has* payment processors.

---

### Phase 3: Change Calculation (20-25 minutes)

**What to do**:
```python
def calculate_change(self, change_amount: float) -> Tuple[bool, Dict[int, int], str]:
    # Convert to cents to avoid floating point errors
    change_cents = int(round(change_amount * 100))
    
    # Greedy algorithm: largest denomination first
    for denom in [2000, 1000, 500, 100, 25, 10, 5, 1]:
        available = self.cash_inventory[denom]
        needed = remaining // denom
        to_give = min(needed, available)
        
        if to_give > 0:
            change_breakdown[denom] = to_give
            remaining -= to_give * denom
    
    # Check if exact change possible
    if remaining > 0:
        return False, {}, "Cannot make exact change"
```

**Why Greedy Algorithm**:
- Works for canonical coin systems (US currency)
- O(n) time complexity where n = number of denominations
- Simple and efficient

**Critical Detail**: Always work in cents to avoid floating point errors!

**When Greedy Fails**: For non-canonical coin systems (e.g., [1, 3, 4]), greedy doesn't give optimal change. Would need dynamic programming.

---

### Phase 4: State Machine (30-40 minutes)

**What to do**:
```python
# State pattern for vending machine lifecycle
class VendingMachineState(ABC):
    def insert_payment(machine, amount, type) -> Tuple[bool, str]
    def select_product(machine, slot_id) -> Tuple[bool, str]
    def cancel(machine) -> Tuple[bool, str]

class IdleState(VendingMachineState):
    # Ready for new transaction
    
class PaymentReceivedState(VendingMachineState):
    # Payment received, waiting for product selection
    
class DispensingState(VendingMachineState):
    # Product and change being dispensed
    
class ErrorState(VendingMachineState):
    # Error occurred, refund initiated
```

**State Transition Flow**:
```
IDLE → insert_payment() → PAYMENT_RECEIVED
PAYMENT_RECEIVED → select_product() → DISPENSING → IDLE
PAYMENT_RECEIVED → cancel() → IDLE
ANY_STATE → error → ERROR → IDLE
```

**Why State Pattern**:
- Each state encapsulates valid operations
- Prevents invalid operations (e.g., selecting product without payment)
- Clear state transitions and responsibilities
- Easy to add new states

**Interview Tip**: Draw the state diagram on whiteboard. Interviewers love visual representation.

---

### Phase 5: Main Controller (40-50 minutes)

**What to do**:
```python
class VendingMachine:
    def __init__(self):
        self.inventory: Dict[str, InventorySlot]
        self.payment_processors: Dict[PaymentType, PaymentProcessor]
        self.current_state: VendingMachineState
        self.transactions: List[Transaction]
        self.lock = RLock()
    
    def insert_payment(self, amount, payment_type):
        with self.lock:
            return self.current_state.insert_payment(self, amount, payment_type)
    
    def _dispense_product(self, slot_id):
        # Critical section: atomic product dispensing
        # 1. Validate payment
        # 2. Calculate change
        # 3. Process payment
        # 4. Dispense product
        # 5. Record transaction
        # 6. Reset state
```

**Critical Operations in _dispense_product**:
1. **Validate** payment is sufficient
2. **Calculate** change (might fail)
3. **Process** payment through processor
4. **Dispense** product (might fail mechanically)
5. **Refund** if anything fails

**Thread Safety**: All public methods use `with self.lock` to ensure atomic operations.

---

### Phase 6: Error Handling (15-20 minutes)

**What to do**:
```python
def _dispense_product(self, slot_id):
    # Check 1: Valid slot?
    if slot_id not in self.inventory:
        self._handle_error(f"Slot {slot_id} not found")
        return False, "Invalid slot"
    
    # Check 2: Can dispense?
    if not slot.can_dispense():
        self._handle_error(f"Slot {slot_id} cannot dispense")
        return False, "Product unavailable"
    
    # Check 3: Sufficient payment?
    if self.current_payment_amount < product.price:
        self._handle_error("Insufficient payment")
        return False, "Insufficient payment"
    
    # Check 4: Can make change?
    if change_due > 0 and payment_type == CASH:
        success, _, msg = cash_processor.calculate_change(change_due)
        if not success:
            self._refund_payment()
            self._handle_error(f"Cannot make change: {msg}")
            return False, "Cannot make exact change"
    
    # Check 5: Mechanical dispensing?
    dispensed_product = slot.dispense()
    if not dispensed_product:
        self._refund_payment()
        self._handle_error("Mechanical dispensing failure")
        return False, "Dispensing failed. Payment refunded."
```

**Error Recovery Strategy**:
- Always refund on failure
- Record failed transactions for audit
- Transition to ERROR state
- Allow recovery to IDLE state

---

## Critical Knowledge Points

### 1. Why State Pattern?

**Without State Pattern**:
```python
def select_product(self, slot_id):
    if self.state == "IDLE":
        return False, "Insert payment first"
    elif self.state == "PAYMENT_RECEIVED":
        return self._dispense_product(slot_id)
    elif self.state == "DISPENSING":
        return False, "Transaction in progress"
    # ... many more if-elif chains
```

**With State Pattern**:
```python
def select_product(self, slot_id):
    return self.current_state.select_product(self, slot_id)
```

**Benefits**:
- Each state is a separate class
- Adding new states doesn't modify existing code (Open/Closed Principle)
- State-specific logic is encapsulated
- Easier to test individual states

---

### 2. Change Calculation Algorithm

**Greedy Algorithm** (for canonical coin systems):
```python
DENOMINATIONS = [2000, 1000, 500, 100, 25, 10, 5, 1]  # cents

for denom in DENOMINATIONS:
    count = min(remaining // denom, available[denom])
    change[denom] = count
    remaining -= count * denom
```

**Time**: O(n) where n = number of denominations  
**Space**: O(n) for change breakdown

**Why it works**: US currency has canonical property - greedy gives optimal solution.

**When it fails**: For denominations [1, 3, 4], to make 6 cents:
- Greedy: 4 + 1 + 1 = 3 coins
- Optimal: 3 + 3 = 2 coins

For non-canonical systems, need **Dynamic Programming**:
```python
def min_coins(amount, denoms):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for denom in denoms:
            if denom <= i:
                dp[i] = min(dp[i], dp[i - denom] + 1)
    
    return dp[amount]
```

**Time**: O(amount × denominations)  
**Space**: O(amount)

---

### 3. Thread Safety Considerations

**Critical Sections**:
1. **Inventory updates** (dispense, restock)
2. **Payment processing** (accept payment, calculate change)
3. **State transitions** (ensure atomic state changes)

**Solution**: Use `RLock` (reentrant lock)
```python
class VendingMachine:
    def __init__(self):
        self.lock = RLock()
    
    def insert_payment(self, amount, payment_type):
        with self.lock:
            # All operations atomic
            return self.current_state.insert_payment(self, amount, payment_type)
```

**Why RLock vs Lock?**
- RLock allows same thread to acquire lock multiple times
- Useful when `insert_payment()` calls `set_state()` which also needs lock

---

### 4. Avoiding Floating Point Errors

**Problem**:
```python
0.1 + 0.2 = 0.30000000000000004  # Not exactly 0.3!
```

**Solution**: Work in cents (integers)
```python
# Convert dollars to cents
change_cents = int(round(change_amount * 100))

# All calculations in cents
remaining = change_cents
for denom in [100, 25, 10, 5, 1]:  # $1, quarter, dime, nickel, penny
    count = remaining // denom
    remaining -= count * denom

# Convert back to dollars for display
return remaining / 100.0
```

---

### 5. Strategy Pattern for Payments

**Why not if-else?**
```python
# Bad approach
def process_payment(self, amount, type):
    if type == "CASH":
        # 50 lines of cash logic
    elif type == "CARD":
        # 40 lines of card logic
    elif type == "DIGITAL":
        # 30 lines of digital logic
```

**Better with Strategy**:
```python
# Each payment type is separate class
processors = {
    PaymentType.CASH: CashProcessor(),
    PaymentType.CARD: CardProcessor(),
    PaymentType.DIGITAL: DigitalProcessor()
}

processor = processors[payment_type]
success, message = processor.process_payment(amount)
```

**Benefits**:
- Single Responsibility: each processor handles one payment type
- Open/Closed: add new payment types without modifying existing code
- Testability: test each payment type independently

---

## Expected Interview Questions & Answers

### Q1: How would you handle concurrent transactions if multiple people try to use the same machine?

**Answer**:
Current implementation uses `RLock` to ensure thread safety at the machine level. However, for **truly concurrent transactions** (multiple people), we need:

1. **Request Queue**:
```python
class VendingMachine:
    def __init__(self):
        self.request_queue = Queue()
        self.worker_thread = Thread(target=self._process_requests)
        self.worker_thread.start()
    
    def insert_payment(self, amount, payment_type):
        # Add to queue instead of processing immediately
        request = {'type': 'PAYMENT', 'amount': amount, 'payment_type': payment_type}
        self.request_queue.put(request)
        return self.request_queue.get()  # Wait for response
```

2. **Session IDs**:
- Assign unique session ID to each customer
- Track session state separately
- Timeout inactive sessions after 60 seconds

3. **Physical constraints**:
- Only one person can physically access machine at a time
- UI should display "In Use" when transaction active

**Follow-up**: In reality, vending machines are **single-user** devices. Concurrency is more relevant for backend systems managing multiple machines.

---

### Q2: How would you design the inventory management for multiple machines in different locations?

**Answer**:
Need a **centralized inventory management system**:

```python
class InventoryManagementSystem:
    def __init__(self):
        self.machines: Dict[str, VendingMachine] = {}
        self.central_inventory: Dict[str, int] = {}  # product_id -> total_quantity
    
    def register_machine(self, machine: VendingMachine):
        self.machines[machine.machine_id] = machine
    
    def get_low_stock_alerts(self) -> List[Alert]:
        alerts = []
        for machine_id, machine in self.machines.items():
            for slot_id, slot in machine.inventory.items():
                if slot.quantity < slot.max_capacity * 0.2:  # Less than 20%
                    alerts.append(Alert(machine_id, slot_id, slot.quantity))
        return alerts
    
    def optimize_restocking_route(self, alerts: List[Alert]) -> List[str]:
        # Traveling salesman problem - find optimal route to restock multiple machines
        # Could use greedy, nearest neighbor, or genetic algorithm
        pass
    
    def predict_restock_schedule(self, machine_id: str) -> datetime:
        # Machine learning: analyze historical sales data
        # Predict when machine will run out of stock
        pass
```

**Key Features**:
1. **Real-time monitoring**: Track inventory levels across all machines
2. **Predictive analytics**: Forecast when restocking needed
3. **Route optimization**: Minimize travel time for restocking personnel
4. **Popular product analysis**: Which products sell fastest at which locations

---

### Q3: How do you handle the case where product gets stuck (mechanical failure)?

**Answer**:
Multi-layered approach:

**1. Detection**:
```python
class InventorySlot:
    def dispense(self) -> Optional[Product]:
        # Decrement quantity
        self.quantity -= 1
        
        # Simulate sensor check
        if not self._sensor_detects_product_dispensed():
            # Product stuck! Rollback
            self.quantity += 1
            self.dispensing_failures += 1
            
            # Mark slot as problematic after 3 failures
            if self.dispensing_failures >= 3:
                self.is_operational = False
            
            return None
        
        return self.product
```

**2. Recovery**:
```python
def _dispense_product(self, slot_id):
    # Try to dispense
    dispensed = slot.dispense()
    
    if not dispensed:
        # Mechanical failure - refund customer
        self._refund_payment()
        
        # Try alternate slot with same product
        alternate_slot = self._find_alternate_slot(slot.product.product_id)
        if alternate_slot:
            return self._dispense_product(alternate_slot)
        
        # No alternate - full refund
        return False, "Product unavailable. Payment refunded."
```

**3. Monitoring**:
- Send alert to maintenance team
- Mark slot as non-operational
- Update inventory system
- Log incident for analysis

**4. Customer Service**:
- Display clear error message
- Provide refund receipt with transaction ID
- Offer customer service phone number
- Option to select different product

---

### Q4: How would you implement a pricing strategy with promotions (e.g., buy 2 get 1 free)?

**Answer**:
Add a **Pricing Strategy** layer:

```python
class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, items: List[Product]) -> float:
        pass

class RegularPricing(PricingStrategy):
    def calculate_price(self, items: List[Product]) -> float:
        return sum(item.price for item in items)

class BuyTwoGetOneFree(PricingStrategy):
    def __init__(self, product_id: str):
        self.product_id = product_id
    
    def calculate_price(self, items: List[Product]) -> float:
        # Count qualifying products
        qualifying = [item for item in items if item.product_id == self.product_id]
        other = [item for item in items if item.product_id != self.product_id]
        
        # Every 3rd item is free
        charged = qualifying[:len(qualifying) - len(qualifying) // 3]
        
        return (sum(item.price for item in charged) + 
                sum(item.price for item in other))

class HappyHourDiscount(PricingStrategy):
    def __init__(self, start_hour: int, end_hour: int, discount: float):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.discount = discount
    
    def calculate_price(self, items: List[Product]) -> float:
        current_hour = datetime.now().hour
        base_price = sum(item.price for item in items)
        
        if self.start_hour <= current_hour <= self.end_hour:
            return base_price * (1 - self.discount)
        return base_price
```

**Integration**:
```python
class VendingMachine:
    def __init__(self):
        self.pricing_strategy: PricingStrategy = RegularPricing()
    
    def set_pricing_strategy(self, strategy: PricingStrategy):
        self.pricing_strategy = strategy
    
    def _dispense_product(self, slot_id):
        # Calculate price using strategy
        final_price = self.pricing_strategy.calculate_price([product])
        
        # Check payment against calculated price
        if self.current_payment_amount < final_price:
            return False, f"Insufficient payment. Need ${final_price:.2f}"
```

---

### Q5: How would you optimize the change-making algorithm if the machine frequently runs out of certain denominations?

**Answer**:
Instead of pure greedy, use **dynamic programming with availability constraints**:

```python
def calculate_optimal_change(self, change_cents: int) -> Optional[Dict[int, int]]:
    """
    Find change combination that:
    1. Minimizes total coins given
    2. Preserves smaller denominations (quarters, dimes, nickels)
    3. Uses larger bills when possible
    """
    # DP state: dp[amount] = (num_coins, denomination_breakdown)
    dp = {0: (0, {})}
    
    for amount in range(1, change_cents + 1):
        best = (float('inf'), {})
        
        for denom in self.DENOMINATIONS:
            # Can we use this denomination?
            if (denom <= amount and 
                amount - denom in dp and 
                self.cash_inventory[denom] > 0):
                
                prev_coins, prev_breakdown = dp[amount - denom]
                new_breakdown = prev_breakdown.copy()
                new_breakdown[denom] = new_breakdown.get(denom, 0) + 1
                
                # Check if we have enough of this denomination
                if new_breakdown[denom] <= self.cash_inventory[denom]:
                    # Apply penalty for using smaller denominations
                    # (encourages using larger bills)
                    penalty = 0
                    if denom <= 25:  # Quarters and smaller
                        penalty = 0.1
                    
                    new_coins = prev_coins + 1 + penalty
                    
                    if new_coins < best[0]:
                        best = (new_coins, new_breakdown)
        
        if best[0] != float('inf'):
            dp[amount] = best
    
    if change_cents in dp:
        _, breakdown = dp[change_cents]
        return breakdown
    return None
```

**Alternative approach**: **Heuristic-based**:
```python
def calculate_smart_change(self, change_cents: int) -> Optional[Dict[int, int]]:
    # Rule 1: If we have plenty of all denominations, use greedy
    if self._has_sufficient_inventory():
        return self._greedy_change(change_cents)
    
    # Rule 2: If low on small denominations, prefer larger bills
    if self._low_on_small_denominations():
        return self._use_larger_bills_first(change_cents)
    
    # Rule 3: Otherwise, use DP
    return self._dp_change(change_cents)
```

---

### Q6: What if a customer disputes a transaction (claims they paid but didn't receive product)?

**Answer**:
Implement comprehensive **audit trail** and **reconciliation**:

```python
@dataclass
class Transaction:
    transaction_id: str
    timestamp: float
    
    # Payment details
    payment_type: PaymentType
    amount_paid: float
    payment_authorization_code: Optional[str]  # From payment processor
    
    # Product details
    product_id: str
    slot_id: str
    product_price: float
    
    # Dispensing details
    dispense_attempted: bool
    dispense_successful: bool
    dispense_sensor_reading: Optional[str]  # From physical sensor
    
    # Change details
    change_due: float
    change_given: Dict[int, int]  # Denomination breakdown
    
    # Status
    status: TransactionStatus
    error_message: Optional[str]
    
    # Audit
    machine_id: str
    user_session_id: Optional[str]
    video_snapshot_id: Optional[str]  # Reference to security camera footage

class VendingMachine:
    def _dispense_product(self, slot_id):
        # Record every step
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            timestamp=time.time(),
            ...
        )
        
        # Step 1: Record payment received
        transaction.payment_authorization_code = payment_processor.auth_code
        
        # Step 2: Record dispense attempt
        transaction.dispense_attempted = True
        dispensed = slot.dispense()
        
        # Step 3: Record sensor reading
        transaction.dispense_sensor_reading = self._read_dispense_sensor()
        transaction.dispense_successful = (dispensed is not None)
        
        # Step 4: Save to persistent storage
        self.transaction_log.save(transaction)
        
        # Step 5: Send to central server
        self.api_client.upload_transaction(transaction)
```

**Dispute Resolution Process**:
1. **Look up transaction** by ID, timestamp, or payment authorization
2. **Check sensor readings** - did product actually dispense?
3. **Review video footage** (if available)
4. **Check inventory** - did slot quantity decrease?
5. **Verify payment** - did payment processor confirm?

**Automatic resolution**:
```python
def investigate_dispute(self, transaction_id: str) -> DisputeResolution:
    transaction = self.transaction_log.get(transaction_id)
    
    # Check 1: Payment confirmed?
    payment_confirmed = self._verify_payment_with_processor(transaction)
    
    # Check 2: Product dispensed according to sensors?
    dispense_confirmed = transaction.dispense_successful
    
    # Check 3: Inventory changed?
    inventory_decreased = self._verify_inventory_change(transaction)
    
    if payment_confirmed and not dispense_confirmed:
        # Customer paid but didn't receive product
        return DisputeResolution(
            outcome="REFUND_CUSTOMER",
            reason="Confirmed dispensing failure",
            refund_amount=transaction.product_price
        )
    
    if payment_confirmed and dispense_confirmed and inventory_decreased:
        # All systems confirm successful transaction
        return DisputeResolution(
            outcome="DENY_DISPUTE",
            reason="All systems confirm successful transaction"
        )
```

---

### Q7: How would you implement a recommender system ("Customers who bought this also bought...")?

**Answer**:
Add **collaborative filtering** layer:

```python
class RecommendationEngine:
    def __init__(self):
        # Transaction history for pattern mining
        self.transaction_history: List[Transaction] = []
        
        # Association rules: product_id -> [frequently bought together]
        self.associations: Dict[str, List[Tuple[str, float]]] = {}
    
    def train(self):
        """
        Build association rules from transaction history
        
        Uses Apriori algorithm to find frequent itemsets
        """
        # Group transactions by session (5-minute window)
        sessions = self._group_by_session()
        
        # Find frequent pairs
        pair_counts = defaultdict(int)
        product_counts = defaultdict(int)
        
        for session in sessions:
            products = [t.product_id for t in session]
            product_counts[products[0]] += 1
            
            # Count co-occurrences in same session
            for i, product_a in enumerate(products):
                for product_b in products[i+1:]:
                    pair = tuple(sorted([product_a, product_b]))
                    pair_counts[pair] += 1
        
        # Calculate confidence: P(B|A) = count(A,B) / count(A)
        for (product_a, product_b), count in pair_counts.items():
            confidence_a_to_b = count / product_counts[product_a]
            confidence_b_to_a = count / product_counts[product_b]
            
            if confidence_a_to_b > 0.1:  # 10% threshold
                self.associations[product_a].append((product_b, confidence_a_to_b))
            
            if confidence_b_to_a > 0.1:
                self.associations[product_b].append((product_a, confidence_b_to_a))
        
        # Sort by confidence
        for product_id in self.associations:
            self.associations[product_id].sort(key=lambda x: x[1], reverse=True)
    
    def get_recommendations(self, product_id: str, top_n: int = 3) -> List[str]:
        """Get top N recommended products for given product"""
        if product_id not in self.associations:
            return []
        
        return [prod_id for prod_id, _ in self.associations[product_id][:top_n]]

# Integration with vending machine
class VendingMachine:
    def __init__(self):
        self.recommender = RecommendationEngine()
    
    def select_product(self, slot_id: str):
        success, message = self._dispense_product(slot_id)
        
        if success:
            # Show recommendations
            product = self.inventory[slot_id].product
            recommendations = self.recommender.get_recommendations(product.product_id)
            
            if recommendations:
                print("\n🌟 Customers also enjoyed:")
                for rec_id in recommendations:
                    rec_slot = self._find_slot_for_product(rec_id)
                    if rec_slot:
                        print(f"   - {rec_slot.product.name} (${rec_slot.product.price:.2f})")
        
        return success, message
```

---

## Testing Strategy

### Unit Tests

**Test payment processors independently**:
```python
def test_cash_processor_exact_change():
    processor = CashProcessor()
    success, change, msg = processor.calculate_change(0.50)
    assert success
    assert 25 in change and change[25] == 2  # 2 quarters

def test_cash_processor_insufficient_denominations():
    processor = CashProcessor()
    # Empty inventory
    processor.cash_inventory = {denom: 0 for denom in processor.DENOMINATIONS}
    success, change, msg = processor.calculate_change(1.00)
    assert not success
    assert "Cannot make exact change" in msg
```

**Test state transitions**:
```python
def test_state_machine_happy_path():
    machine = VendingMachine("VM001")
    machine.add_inventory_slot(InventorySlot("A1", Product("P001", "Snack", 1.50, "Snacks"), 5, 10))
    
    # Initial state
    assert machine.current_state.get_state_name() == "IDLE"
    
    # Insert payment
    machine.insert_payment(2.00, PaymentType.CASH)
    assert machine.current_state.get_state_name() == "PAYMENT_RECEIVED"
    
    # Select product
    machine.select_product("A1")
    assert machine.current_state.get_state_name() == "IDLE"

def test_state_machine_cancellation():
    machine = VendingMachine("VM001")
    machine.insert_payment(2.00, PaymentType.CASH)
    assert machine.current_state.get_state_name() == "PAYMENT_RECEIVED"
    
    machine.cancel_transaction()
    assert machine.current_state.get_state_name() == "IDLE"
    assert machine.current_payment_amount == 0
```

---

### Integration Tests

**Test full transaction flow**:
```python
def test_complete_cash_transaction_with_change():
    machine = VendingMachine("VM001")
    machine.add_inventory_slot(InventorySlot("A1", Product("P001", "Soda", 1.50, "Beverages"), 10, 15))
    
    initial_inventory = machine.inventory["A1"].quantity
    
    # Complete transaction
    machine.insert_payment(5.00, PaymentType.CASH)
    success, message = machine.select_product("A1")
    
    assert success
    assert "Change:" in message
    assert machine.inventory["A1"].quantity == initial_inventory - 1
    assert len(machine.transactions) == 1
    assert machine.transactions[0].status == TransactionStatus.COMPLETED

def test_mechanical_failure_recovery():
    machine = VendingMachine("VM001")
    
    # Create slot that will fail to dispense
    failing_slot = InventorySlot("A1", Product("P001", "Snack", 1.50, "Snacks"), 5, 10)
    failing_slot.is_operational = False
    machine.add_inventory_slot(failing_slot)
    
    machine.insert_payment(1.50, PaymentType.CASH)
    success, message = machine.select_product("A1")
    
    assert not success
    assert "unavailable" in message.lower()
    # Check refund occurred
    assert len([t for t in machine.transactions if t.status == TransactionStatus.REFUNDED]) > 0
```

---

### Load Testing

**Concurrent transaction simulation**:
```python
import threading

def test_concurrent_transactions():
    machine = VendingMachine("VM001")
    machine.add_inventory_slot(InventorySlot("A1", Product("P001", "Snack", 1.00, "Snacks"), 100, 100))
    
    successful_transactions = []
    
    def make_purchase():
        try:
            machine.insert_payment(1.00, PaymentType.CARD)
            success, msg = machine.select_product("A1")
            if success:
                successful_transactions.append(True)
        except Exception as e:
            print(f"Error: {e}")
    
    # Simulate 50 concurrent purchases
    threads = [threading.Thread(target=make_purchase) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Due to thread safety, some transactions might fail, but no corruption
    assert machine.inventory["A1"].quantity >= 50
    assert len(machine.transactions) > 0
```

---

## Production Considerations

### 1. Persistence

**Current implementation**: In-memory only  
**Production needs**: Database persistence

```python
class VendingMachine:
    def __init__(self, machine_id: str, db: Database):
        self.db = db
        
        # Load state from database
        self._load_inventory()
        self._load_transactions()
    
    def _load_inventory(self):
        rows = self.db.query("SELECT * FROM inventory WHERE machine_id = ?", self.machine_id)
        for row in rows:
            slot = InventorySlot(
                slot_id=row['slot_id'],
                product=self._get_product(row['product_id']),
                quantity=row['quantity'],
                max_capacity=row['max_capacity']
            )
            self.inventory[slot.slot_id] = slot
    
    def _dispense_product(self, slot_id):
        # ... dispensing logic ...
        
        # Persist transaction
        self.db.execute("""
            INSERT INTO transactions (id, machine_id, product_id, amount_paid, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, transaction.transaction_id, self.machine_id, transaction.product_id, 
           transaction.amount_paid, transaction.status.value, transaction.timestamp)
        
        # Update inventory
        self.db.execute("""
            UPDATE inventory SET quantity = quantity - 1
            WHERE machine_id = ? AND slot_id = ?
        """, self.machine_id, slot_id)
```

---

### 2. Monitoring & Alerts

**Implement health checks**:
```python
class HealthMonitor:
    def check_machine_health(self, machine: VendingMachine) -> HealthReport:
        issues = []
        
        # Check 1: Low inventory
        for slot_id, slot in machine.inventory.items():
            if slot.quantity < slot.max_capacity * 0.2:
                issues.append(f"Low stock in {slot_id}: {slot.quantity} remaining")
        
        # Check 2: Low cash for change
        cash_processor = machine.payment_processors[PaymentType.CASH]
        if cash_processor.get_total_cash() < 20.00:
            issues.append(f"Low cash reserves: ${cash_processor.get_total_cash():.2f}")
        
        # Check 3: Non-operational slots
        non_op = [slot_id for slot_id, slot in machine.inventory.items() if not slot.is_operational]
        if non_op:
            issues.append(f"Non-operational slots: {', '.join(non_op)}")
        
        # Check 4: High failure rate
        recent_transactions = machine.transactions[-100:]
        failed = len([t for t in recent_transactions if t.status == TransactionStatus.FAILED])
        if failed > 10:  # More than 10% failure rate
            issues.append(f"High failure rate: {failed}% of recent transactions")
        
        return HealthReport(
            machine_id=machine.machine_id,
            status="HEALTHY" if not issues else "DEGRADED" if len(issues) < 3 else "CRITICAL",
            issues=issues,
            timestamp=time.time()
        )
```

---

### 3. Security

**Key concerns**:
1. **Payment fraud**: Secure communication with payment processors
2. **Physical tampering**: Detect unauthorized access to cash box
3. **Data privacy**: Encrypt transaction data
4. **Audit trail**: Immutable transaction logs

```python
class SecureVendingMachine(VendingMachine):
    def __init__(self, machine_id: str, encryption_key: bytes):
        super().__init__(machine_id)
        self.encryption_key = encryption_key
        self.tamper_sensor = TamperSensor()
    
    def _save_transaction(self, transaction: Transaction):
        # Encrypt sensitive data
        encrypted_data = self._encrypt(transaction.to_json())
        
        # Hash for integrity
        transaction_hash = hashlib.sha256(encrypted_data).hexdigest()
        
        # Store with signature
        self.db.execute("""
            INSERT INTO transactions (id, data, hash, timestamp)
            VALUES (?, ?, ?, ?)
        """, transaction.transaction_id, encrypted_data, transaction_hash, time.time())
    
    def check_tamper(self):
        if self.tamper_sensor.is_tampered():
            # Alert security
            self.alert_system.send_alert("TAMPER_DETECTED", self.machine_id)
            
            # Lock machine
            self.set_state(self.error_state)
            
            # Log incident
            self.security_log.log("Tamper detected", severity="CRITICAL")
```

---

### 4. Scalability

**Managing fleet of machines**:
```python
class VendingMachineFleet:
    def __init__(self):
        self.machines: Dict[str, VendingMachine] = {}
        self.message_queue = MessageQueue()  # RabbitMQ, Kafka, etc.
    
    def register_machine(self, machine: VendingMachine):
        self.machines[machine.machine_id] = machine
        
        # Subscribe to machine events
        self.message_queue.subscribe(f"machine.{machine.machine_id}.events", 
                                     self._handle_machine_event)
    
    def _handle_machine_event(self, event: dict):
        event_type = event['type']
        machine_id = event['machine_id']
        
        if event_type == "LOW_INVENTORY":
            self._schedule_restock(machine_id)
        elif event_type == "MECHANICAL_FAILURE":
            self._dispatch_technician(machine_id)
        elif event_type == "HIGH_SALES":
            self._analyze_demand_surge(machine_id)
    
    def get_analytics(self) -> FleetAnalytics:
        return FleetAnalytics(
            total_machines=len(self.machines),
            total_revenue=sum(m.get_sales_summary()['total_revenue'] for m in self.machines.values()),
            avg_uptime=self._calculate_avg_uptime(),
            most_popular_products=self._aggregate_product_sales(),
            machines_needing_attention=self._get_unhealthy_machines()
        )
```

---

## Summary

### Do's ✅
- Use **State Pattern** for clear state management
- Use **Strategy Pattern** for payment processing
- Work in **cents** to avoid floating point errors
- Implement comprehensive **error handling and refunds**
- Use **RLock** for thread safety
- Maintain detailed **transaction audit trails**
- Plan for **mechanical failures** and recovery

### Don'ts ❌
- Don't use floating point for money calculations
- Don't assume greedy algorithm works for all coin systems
- Don't forget to refund on errors
- Don't ignore thread safety in multi-threaded environments
- Don't skip state validation (e.g., selecting product without payment)
- Don't overlook physical constraints (slot capacity, change availability)

### Key Takeaways
1. **State Pattern**: Encapsulates state-specific behavior, prevents invalid operations
2. **Payment Strategies**: Each payment type has unique processing logic
3. **Atomicity**: Dispense operation must be atomic - all or nothing
4. **Error Recovery**: Always refund on failure, maintain audit trail
5. **Production Ready**: Requires persistence, monitoring, security, scalability

---

**Time to Master**: 4-6 hours  
**Difficulty**: Medium  
**Key Patterns**: State, Strategy  
**Critical Skills**: State machines, payment processing, error handling, thread safety

