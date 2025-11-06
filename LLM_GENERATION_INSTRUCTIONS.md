# LLM Instructions for Generating LLD Interview Prep Files

## Overview
You are tasked with creating high-quality Low-Level Design (LLD) interview preparation materials for a software engineer with 4 years of experience. You will generate TWO files for each problem:
1. A Python implementation file (`XX_topic_name.py`)
2. A comprehensive README file (`XX_topic_name_readme.md`)

You will also update the README.md file following the same style as done in the existing README.md file.

**THE USER WILL PASTE THE PROBLEM STATEMENT AT THE TOP OF THIS FILE. READ IT AND GENERATE BOTH FILES.**

---

## Quality Standards

### Reference Files to Study First
Before generating, examine these reference files to understand the expected quality:
- `01_cache_system.py` - Gold standard for code structure
- `01_cache_system_readme.md` - Gold standard for README structure
- `04_elevator_system.py` - Example of state machine implementation
- `05_vending_machine.py` - Example of strategy + state patterns

### Key Principles
- **Interview-focused**: Every comment and section should help with interview preparation
- **Practical, not academic**: Focus on real implementation challenges
- **Clear structure**: Visual separators, section headers, organized code
- **Deep explanations**: Why this approach? What are alternatives? What are trade-offs?
- **Experience-appropriate**: Suitable for 4 years experience (not junior, not senior)

---

## PART 1: Python Implementation File (`XX_topic_name.py`)

### File Structure Template

```python
"""
[Problem Name] - Single File Implementation
For coding interviews and production-ready reference

Features:
- [Feature 1]
- [Feature 2]
- [Feature 3]
- [Feature 4]

Interview Focus:
- [Design pattern 1]
- [Design pattern 2]
- [Key algorithm/technique]
- [Concurrency consideration]
- [Edge case handling]

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
# SECTION 1: MODELS - Core data classes and enums
# ============================================================================

class SomeEnum(Enum):
    """Brief description of enum purpose"""
    VALUE1 = "value1"
    VALUE2 = "value2"


@dataclass
class CoreEntity:
    """
    Description of entity
    
    Interview Focus: Why this structure? What are alternatives?
    """
    field1: str
    field2: int
    
    def some_method(self) -> bool:
        """
        Description
        
        Key Insight: Important detail about this method
        """
        pass


# ============================================================================
# SECTION 2: [PATTERN/COMPONENT NAME] - Brief description
# ============================================================================

class AbstractBase(ABC):
    """
    Description of abstract class
    
    [Pattern Name] Pattern: Why we use this pattern
    
    Interview Focus: How does this pattern help?
    """
    
    @abstractmethod
    def key_method(self, param: Type) -> Type:
        """Method description"""
        pass


class ConcreteImplementation(AbstractBase):
    """
    Concrete implementation description
    
    Key Features:
    - Feature 1
    - Feature 2
    
    Interview Focus: What makes this implementation special?
    """
    
    def __init__(self):
        self.lock = RLock()
    
    def key_method(self, param: Type) -> Type:
        """
        Implementation description
        
        Time Complexity: O(?)
        Space Complexity: O(?)
        
        Key Insight: Important algorithmic detail
        """
        with self.lock:
            # Implementation with detailed comments
            pass


# ============================================================================
# SECTION 3: MAIN CONTROLLER - Core system logic
# ============================================================================

class MainSystem:
    """
    Main system controller
    
    Responsibilities:
    - Responsibility 1
    - Responsibility 2
    - Responsibility 3
    
    Thread Safety: Uses RLock for all operations
    
    Interview Focus: How do you coordinate multiple components?
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        
        # Component initialization with comments
        self.component1 = Component1()
        self.component2 = Component2()
        
        # State tracking
        self.current_state = InitialState()
        
        # Thread safety
        self.lock = RLock()
    
    def public_method(self, param: Type) -> Tuple[bool, str]:
        """
        Public API method
        
        Interview Focus: How do you design public APIs?
        """
        with self.lock:
            return self._internal_method(param)
    
    def _internal_method(self, param: Type) -> Tuple[bool, str]:
        """
        Internal implementation
        
        Interview Focus: How do you handle the critical operation?
        
        Key Challenges:
        - Challenge 1
        - Challenge 2
        - Challenge 3
        """
        # Detailed implementation with error handling
        pass


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
    """Demonstrate basic system operations"""
    print_separator("Basic Operations")
    
    # Initialize system
    system = MainSystem("SYS001")
    
    # Demonstrate basic flow
    print("\n🔹 Test 1: Basic operation")
    # Test code with clear output
    pass


def demo_advanced_features():
    """Demonstrate advanced features"""
    print_separator("Advanced Features")
    
    # Demonstrate complex scenarios
    pass


def demo_error_handling():
    """Demonstrate error handling"""
    print_separator("Error Handling")
    
    # Demonstrate error scenarios
    pass


def demo_concurrency():
    """Demonstrate thread safety"""
    print_separator("Concurrency Handling")
    
    # Demonstrate concurrent operations
    pass


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  [SYSTEM NAME] - COMPREHENSIVE DEMONSTRATION")
    print("  Features: [List key features]")
    print("="*70)
    
    demo_basic_operations()
    demo_advanced_features()
    demo_error_handling()
    demo_concurrency()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python XX_topic_name.py
    """
    run_demo()
```

### Critical Code Requirements

1. **Section Headers**: Use `# ============================================================================`
2. **Comments Style**:
   - Docstrings for all classes and methods
   - "Interview Focus:" annotations for key concepts
   - "Key Insight:" for algorithmic details
   - Time/Space complexity for algorithms
   - "Key Challenges:" for complex operations

3. **Code Organization**:
   - Imports at top (abc, dataclasses, typing, threading, etc.)
   - Models section first (enums, dataclasses)
   - Pattern implementations (Strategy, State, etc.)
   - Main controller class
   - Demo functions at end

4. **Design Patterns**: Use appropriate patterns (State, Strategy, Factory, Observer, etc.)

5. **Thread Safety**: Use `RLock` for concurrent operations, always `with self.lock:`

6. **Error Handling**: Return `Tuple[bool, str]` for operations that can fail

7. **Demo Functions**:
   - `demo_basic_operations()` - Happy path
   - `demo_advanced_features()` - Complex scenarios
   - `demo_error_handling()` - Error cases
   - `demo_concurrency()` - Thread safety (if applicable)
   - `run_demo()` - Main orchestrator

---

## PART 2: README File (`XX_topic_name_readme.md`)

### README Structure Template

```markdown
# [Problem Name] - Interview Preparation Guide

**Target Audience**: Software Engineers with 2-5 years of experience  
**Focus**: [Core design concepts]  
**Estimated Study Time**: X-Y hours

Make sure the readme file is not too long and boring.

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

Design a [system name] that can:
- [Requirement 1]
- [Requirement 2]
- [Requirement 3]
- [Requirement 4]

**Core Challenge**: [What's the hardest part about this problem?]

---

## Step-by-Step Implementation Guide

### Phase 1: [First Component] (X-Y minutes)

**What to do**:
```python
# Show minimal code structure
class Component:
    key_fields
    key_methods
```

**Why this approach**:
- Reason 1
- Reason 2
- Reason 3

**Common mistake**: [What do people get wrong here?]

---

### Phase 2: [Second Component] (X-Y minutes)

**What to do**:
```python
# Code structure
```

**Why [Pattern Name] Pattern**:
- Benefit 1
- Benefit 2
- Benefit 3

**Interview Insight**: [Key insight for interviews]

---

### Phase 3: [Algorithm/Logic] (X-Y minutes)

**What to do**:
```python
def key_algorithm(params):
    # Core logic with comments
```

**Why [Algorithm Choice]**:
- Performance characteristic
- Trade-offs
- Alternatives

**Critical Detail**: [Important implementation detail]

**When it fails**: [Edge cases or limitations]

---

### Phase 4: [Integration] (X-Y minutes)

**What to do**:
```python
# Integration code
```

**State Transition Flow** (if applicable):
```
STATE1 → action() → STATE2
STATE2 → action() → STATE3
```

**Why this pattern**:
- Benefit 1
- Benefit 2

**Interview Tip**: [Practical interview advice]

---

### Phase 5: [Error Handling] (X-Y minutes)

**What to do**:
```python
# Error handling pattern
```

**Error Recovery Strategy**:
- Strategy point 1
- Strategy point 2

---

## Critical Knowledge Points

### 1. Why [Design Pattern]?

**Without Pattern**:
```python
# Show problematic code
```

**With Pattern**:
```python
# Show clean code
```

**Benefits**:
- Benefit 1 with explanation
- Benefit 2 with explanation

---

### 2. [Key Algorithm] Explained

**Algorithm**:
```python
# Core algorithm with detailed comments
```

**Time**: O(?)  
**Space**: O(?)

**Why it works**: [Explanation]

**When it fails**: [Edge cases]

**Alternative approach**:
```python
# Alternative algorithm
```

**Time**: O(?)  
**Space**: O(?)

---

### 3. Thread Safety Considerations

**Critical Sections**:
1. Section 1
2. Section 2
3. Section 3

**Solution**: [Threading approach]
```python
# Threading code example
```

**Why RLock vs Lock?** [Explanation]

---

### 4. [Domain-Specific Challenge]

**Problem**: [What's the challenge?]

**Solution**: [How to solve it]
```python
# Solution code
```

---

## Expected Interview Questions & Answers

### Q1: [Fundamental scaling question]

**Answer**:
[Multi-paragraph detailed answer explaining the approach]

**Implementation**:
```python
# Code showing the solution
```

**Key Points**:
1. Point 1
2. Point 2
3. Point 3

**Follow-up**: [Common follow-up question and brief answer]

---

### Q2: [Pattern/Design question]

**Answer**:
[Detailed answer with code examples]

**Benefits**:
1. Benefit 1
2. Benefit 2

---

### Q3: [Error handling question]

**Answer**:
Multi-layered approach:

**1. Detection**:
```python
# Detection code
```

**2. Recovery**:
```python
# Recovery code
```

**3. Monitoring**:
- Monitor aspect 1
- Monitor aspect 2

---

### Q4: [Extension/Feature question]

**Answer**:
Add a [new component] layer:

```python
# New component implementation
```

**Integration**:
```python
# How to integrate with existing system
```

---

### Q5: [Optimization question]

**Answer**:
[Explanation of optimization approach]

```python
# Optimized implementation
```

**Alternative approach**: [Another way to solve it]

---

### Q6: [Real-world scenario question]

**Answer**:
[Detailed answer with production considerations]

**Implementation**:
```python
# Production-ready code
```

---

### Q7: [Advanced feature question]

**Answer**:
[Comprehensive answer with code]

---

## Testing Strategy

### Unit Tests

**Test [component] independently**:
```python
def test_basic_functionality():
    # Test code
    assert condition

def test_edge_case():
    # Edge case test
    assert condition
```

**Test [pattern] behavior**:
```python
def test_state_transitions():
    # State test code
```

---

### Integration Tests

**Test full flow**:
```python
def test_complete_operation():
    # Integration test
```

**Test error recovery**:
```python
def test_error_handling():
    # Error test
```

---

### Load Testing

**Concurrent operation simulation**:
```python
import threading

def test_concurrent_operations():
    # Concurrency test
```

---

## Production Considerations

### 1. Persistence

**Current implementation**: In-memory only  
**Production needs**: Database persistence

```python
class ProductionSystem:
    def __init__(self, system_id: str, db: Database):
        # Load from database
        self._load_state()
    
    def _load_state(self):
        # Database loading code
```

---

### 2. Monitoring & Alerts

**Implement health checks**:
```python
class HealthMonitor:
    def check_system_health(self, system) -> HealthReport:
        # Health check implementation
```

---

### 3. Security

**Key concerns**:
1. Concern 1
2. Concern 2
3. Concern 3

```python
class SecureSystem(MainSystem):
    # Security implementation
```

---

### 4. Scalability

**Managing distributed systems**:
```python
class DistributedSystem:
    # Distributed system implementation
```

---

## Summary

### Do's ✅
- Do item 1
- Do item 2
- Do item 3
- Do item 4
- Do item 5

### Don'ts ❌
- Don't item 1
- Don't item 2
- Don't item 3
- Don't item 4

### Key Takeaways
1. **Takeaway 1**: Explanation
2. **Takeaway 2**: Explanation
3. **Takeaway 3**: Explanation
4. **Takeaway 4**: Explanation
5. **Takeaway 5**: Explanation

---

**Time to Master**: X-Y hours  
**Difficulty**: [Easy/Medium/Hard]  
**Key Patterns**: [Pattern1, Pattern2]  
**Critical Skills**: [Skill1, skill2, skill3]
```

### Critical README Requirements

1. **Structure**: Follow exact table of contents order
2. **Phase Breakdown**: 5-6 phases with time estimates (15-50 minutes each)
3. **Code Examples**: Include code in every section, not just explanations
4. **Questions**: 5-7 detailed interview and follow-up questions with comprehensive answers
5. **Depth**: Each Q&A should be 2-4 paragraphs with code examples
6. **Testing**: Show actual test code, not just descriptions
7. **Production**: Show real production concerns with code examples
8. **Summary**: Clear Do's/Don'ts and key takeaways

---

## Generation Workflow

When the user pastes a problem statement at the top of this file:

### Step 1: Analysis (Think before coding)
- Identify core entities and relationships
- Determine appropriate design patterns
- Identify key algorithms needed
- Consider concurrency requirements
- Think about error scenarios

### Step 2: Generate Python File
- Follow the template structure exactly
- Use clear section headers with `=` separators
- Add "Interview Focus" comments throughout
- Implement 4-5 demo functions
- Include comprehensive docstrings
- Add time/space complexity annotations
- Ensure thread safety with RLock

### Step 3: Generate README File
- Follow the template structure exactly
- Break implementation into 5-6 phases
- Include code examples in every section
- Write 5-7 detailed interview Q&As
- Each Q&A must have code examples
- Include testing code examples
- Include production code examples
- End with clear Do's/Don'ts

### Step 4: Quality Check
- Does code follow the structure of `01_cache_system.py`?
- Does README follow the structure of `01_cache_system_readme.md`?
- Are there "Interview Focus" comments throughout code?
- Are there 5-7 detailed Q&As with code in README?
- Is every major section explained with "Why" not just "What"?
- Are there concrete code examples, not just descriptions?
- Is the content appropriate for 4 years experience level?

---

## Examples of Good vs Bad Content

### ❌ BAD (Too Brief, No Depth)
```markdown
### Q1: How do you handle concurrency?
**Answer**: Use locks to prevent race conditions.
```

### ✅ GOOD (Detailed, With Code, Explains Why, but not too deep)
```markdown
### Q1: How would you handle concurrent access to shared inventory?

**Answer**:
The system uses `RLock` (reentrant lock) to ensure thread-safe operations. We need reentrant locks because public methods may call other public methods, and a regular lock would cause deadlock.

**Critical Sections**:
1. **Inventory updates** (add, remove, check availability)
2. **State transitions** (ensure atomic state changes)
3. **Transaction recording** (prevent race conditions on transaction log)

**Implementation**:
```python
class InventoryManager:
    def __init__(self):
        self.lock = RLock()  # Reentrant for nested calls
        self.inventory = {}
    
    def reserve_item(self, item_id: str) -> bool:
        with self.lock:
            if not self._check_availability(item_id):  # Nested call
                return False
            self._decrement_count(item_id)  # Another nested call
            return True
    
    def _check_availability(self, item_id: str) -> bool:
        with self.lock:  # Can reacquire lock
            return self.inventory.get(item_id, 0) > 0
```

**Why RLock vs Lock?**
- Regular `Lock`: Same thread cannot acquire twice → deadlock
- `RLock`: Same thread can acquire multiple times → safe for nested calls

**Alternative approach**: Lock-free data structures using atomic operations, but adds complexity.
```

---

## Common Mistakes to Avoid

1. **Too much code in README**: Show snippets, not full implementations
2. **Not enough "why"**: Always explain why you chose this approach
3. **Missing Interview Focus**: Code should teach interview skills
4. **Weak Q&As**: Questions need detailed, multi-paragraph answers with code
5. **No trade-offs**: Always discuss alternatives and trade-offs
6. **Missing complexity analysis**: Include O(n) time/space for algorithms
7. **No concrete examples**: Every concept needs code examples
8. **Too academic**: Focus on practical interview scenarios
9. **Inconsistent structure**: Follow templates exactly
10. **Missing demo functions**: Must have runnable demonstrations

---

## Final Checklist Before Delivery

### Python File (.py)
- [ ] Uses section headers with `=` separators
- [ ] Has "Interview Focus" comments throughout
- [ ] Includes docstrings for all classes/methods
- [ ] Has time/space complexity for algorithms
- [ ] Uses RLock for thread safety
- [ ] Returns `Tuple[bool, str]` for operations
- [ ] Has 4-5 demo functions
- [ ] Has `run_demo()` main function
- [ ] Implements appropriate design patterns
- [ ] Has detailed inline comments

### README File (.md)
- [ ] Follows exact template structure
- [ ] Has 5-6 implementation phases with time estimates
- [ ] Has 5-7 detailed interview Q&As
- [ ] Every Q&A has code examples
- [ ] Every major section has code examples
- [ ] Explains "why" not just "what"
- [ ] Includes testing code examples
- [ ] Includes production considerations
- [ ] Has Do's/Don'ts section
- [ ] Has clear summary with key takeaways
- [ ] Appropriate for 4 years experience

---

PROBLEM STATEMENT - Design a File System (e.g., Simple Directory Structure)
Focus: Hierarchical directories, file operations (create/delete/rename), permissions, and traversal efficiency.
---
