# Snake and Ladder Game - Low Level Design

## Problem Statement

Design and implement a **Snake and Ladder Game** system that supports:

1. **Classic board game mechanics** with configurable board sizes
2. **Multiple players** with turn-based gameplay
3. **Snakes and ladders** placement with validation
4. **Flexible dice rolling strategies** (standard, weighted, controlled for testing)
5. **Game state management** and move history tracking
6. **Statistics tracking** (snakes hit, ladders climbed, move counts)
7. **Thread-safe operations** for concurrent access
8. **Extensible design** for future features (power-ups, tournaments, AI players)

### Core Requirements

- Support 2+ players with unique identifiers
- Classic 100-cell board with customizable snake/ladder placements
- Dice rolling with multiple strategies (Strategy pattern)
- Automatic win detection when reaching final cell
- Move validation (exact roll to win, boundary checks)
- Complete audit trail of all moves
- Player statistics and leaderboard

### Technical Constraints

- Python 3.8+ (standard library only)
- O(1) position lookups for snakes/ladders
- Thread-safe for multiplayer scenarios
- Memory efficient (no large 2D arrays)
- Testable design with controlled dice

---

## Step-by-Step Implementation Guide

### Phase 1: Core Models and Enums (15 minutes)

**What to build:**
- Enums for cell types and player status
- Dataclasses for Snake, Ladder, Player, GameMove
- Foundation for type safety and clear data contracts

**Interview Focus:**
- Why use enums vs strings? (Type safety, autocomplete, invalid state prevention)
- Why dataclasses? (Reduce boilerplate, automatic __init__/__repr__, immutability with frozen=True)
- Separation of data models from business logic

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

class CellType(Enum):
    """Types of special cells on the board"""
    NORMAL = "normal"
    SNAKE = "snake"
    LADDER = "ladder"
    
class PlayerStatus(Enum):
    """Player game states"""
    WAITING = "waiting"
    PLAYING = "playing"
    WON = "won"

@dataclass
class Snake:
    """Immutable snake representation"""
    head: int  # Starting position
    tail: int  # Ending position (must be < head)
    
    def __post_init__(self):
        if self.tail >= self.head:
            raise ValueError(f"Snake tail {self.tail} must be < head {self.head}")

@dataclass
class Ladder:
    """Immutable ladder representation"""
    bottom: int  # Starting position
    top: int     # Ending position (must be > bottom)
    
    def __post_init__(self):
        if self.top <= self.bottom:
            raise ValueError(f"Ladder top {self.top} must be > bottom {self.bottom}")

@dataclass
class Player:
    """Mutable player with game statistics"""
    id: str
    name: str
    position: int = 0
    status: PlayerStatus = PlayerStatus.WAITING
    moves_count: int = 0
    snakes_hit: int = 0
    ladders_climbed: int = 0
    
@dataclass
class GameMove:
    """Audit trail for each move"""
    player_id: str
    dice_roll: int
    from_position: int
    to_position: int
    cell_type: CellType
    timestamp: float = field(default_factory=time.time)
```

**Key Design Decisions:**
- Snakes/Ladders are **immutable** (frozen-like) - validated in `__post_init__`
- Player is **mutable** - statistics updated during gameplay
- GameMove captures complete context for replay/analysis
- Timestamp enables chronological sorting and time-based analysis

---

### Phase 2: Dice Strategy Pattern (20 minutes)

**What to build:**
- Abstract DiceStrategy base class
- StandardDice (classic 1-6 random)
- WeightedDice (biased probabilities for testing)
- ControlledDice (predetermined sequence for unit tests)

**Interview Focus:**
- Why Strategy pattern vs just `random.randint(1, 6)`?
  - **Testability**: Inject ControlledDice for deterministic tests
  - **Extensibility**: Easy to add loaded dice, N-sided dice, special game modes
  - **Open/Closed Principle**: Add new strategies without modifying game logic
- How does this help in interviews? Shows understanding of SOLID principles

```python
from abc import ABC, abstractmethod
import random
from collections import deque
from typing import List

class DiceStrategy(ABC):
    """Abstract strategy for dice rolling"""
    
    @abstractmethod
    def roll(self) -> int:
        """Roll the dice and return result"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Describe the dice behavior"""
        pass

class StandardDice(DiceStrategy):
    """Classic 6-sided fair dice"""
    
    def __init__(self, sides: int = 6):
        self.sides = sides
    
    def roll(self) -> int:
        return random.randint(1, self.sides)
    
    def get_description(self) -> str:
        return f"Standard {self.sides}-sided dice"

class WeightedDice(DiceStrategy):
    """Dice with biased probabilities"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        # Default: favor high rolls (5-6)
        self.weights = weights or [0.1, 0.1, 0.15, 0.15, 0.25, 0.25]
        self.sides = len(self.weights)
        
    def roll(self) -> int:
        return random.choices(range(1, self.sides + 1), weights=self.weights)[0]
    
    def get_description(self) -> str:
        return f"Weighted {self.sides}-sided dice"

class ControlledDice(DiceStrategy):
    """Predetermined dice sequence for testing"""
    
    def __init__(self, rolls: List[int]):
        if not rolls:
            raise ValueError("Must provide at least one roll")
        self.rolls = deque(rolls)
        self.original_rolls = rolls.copy()
    
    def roll(self) -> int:
        if not self.rolls:
            raise RuntimeError("All predetermined rolls exhausted")
        return self.rolls.popleft()
    
    def get_description(self) -> str:
        return f"Controlled dice: {self.original_rolls}"
```

**Testing Benefits:**
```python
# Unit test example
def test_exact_win():
    game = SnakeAndLadderGame(board_size=20)
    game.set_dice_strategy(ControlledDice([18, 2]))  # 0→18, 18→20 (exact win)
    game.add_player("test")
    game.start_game()
    
    # First roll: 0 + 18 = 18
    move1 = game.roll_dice_and_move()
    assert move1.to_position == 18
    
    # Second roll: 18 + 2 = 20 (wins with exact roll)
    move2 = game.roll_dice_and_move()
    assert move2.to_position == 20
    assert game.get_winner().id == "test"
```

---

### Phase 3: Board Management (25 minutes)

**What to build:**
- Board class with snake/ladder storage
- O(1) position lookup using dictionaries
- Overlap validation (no multiple snakes/ladders at same cell)
- `get_next_position()` with boundary checks

**Interview Focus:**
- **Data structure choice**: Why `dict` instead of 2D array?
  - Sparse data: 10-20 snakes/ladders on 100-cell board
  - O(1) lookup vs O(n) scan
  - Memory efficient (only stores special cells)
- **Validation strategy**: Prevent invalid board states at construction time
- **Boundary handling**: Exact roll required to win (roll 6 at position 95 on 100-cell board → stays at 95)

```python
from typing import Dict

class Board:
    """Game board with snakes and ladders"""
    
    # Interview Focus: Discuss data structure choice
    def __init__(self, size: int = 100):
        if size < 10:
            raise ValueError("Board size must be at least 10")
        
        self.size = size
        self._snakes: Dict[int, Snake] = {}      # position → Snake
        self._ladders: Dict[int, Ladder] = {}    # position → Ladder
        
        # O(1) lookup: position → next_position
        self._position_map: Dict[int, int] = {}
    
    def add_snake(self, head: int, tail: int) -> None:
        """Add a snake with validation"""
        # Boundary checks
        if not (1 <= tail < head <= self.size):
            raise ValueError(f"Invalid snake: {head}→{tail}")
        
        # Overlap check: no snake/ladder at this position already
        if head in self._position_map:
            raise ValueError(f"Position {head} already occupied")
        
        snake = Snake(head, tail)
        self._snakes[head] = snake
        self._position_map[head] = tail
    
    def add_ladder(self, bottom: int, top: int) -> None:
        """Add a ladder with validation"""
        if not (1 <= bottom < top <= self.size):
            raise ValueError(f"Invalid ladder: {bottom}→{top}")
        
        if bottom in self._position_map:
            raise ValueError(f"Position {bottom} already occupied")
        
        ladder = Ladder(bottom, top)
        self._ladders[bottom] = ladder
        self._position_map[bottom] = top
    
    def get_next_position(self, current: int, dice_roll: int) -> tuple[int, CellType]:
        """
        Calculate next position with boundary checks
        
        Interview Focus: Explain boundary logic
        - If roll overshoots, stay at current position
        - This prevents players from "accidentally" winning
        """
        target = current + dice_roll
        
        # Boundary check: can't overshoot final cell
        if target > self.size:
            return current, CellType.NORMAL  # Stay at current
        
        # Check for snake/ladder at target position
        if target in self._position_map:
            final = self._position_map[target]
            cell_type = CellType.SNAKE if target in self._snakes else CellType.LADDER
            return final, cell_type
        
        return target, CellType.NORMAL
    
    def is_winning_position(self, position: int) -> bool:
        """Check if player reached final cell"""
        return position == self.size
```

**Complexity Analysis:**
- `add_snake/add_ladder`: O(1) - dict insertion
- `get_next_position`: O(1) - dict lookup
- Space: O(S + L) where S = snakes, L = ladders (typically 10-20 each)

**Alternative approach (2D array):**
```python
# Less efficient for sparse data
board = [[None] * 10 for _ in range(10)]  # 100 cells
# Requires O(√N) to convert position to (row, col)
# Wastes memory storing empty cells
```

---

### Phase 4: Game Controller (30 minutes)

**What to build:**
- SnakeAndLadderGame main controller
- Player management (add, turn rotation)
- Core game loop (`roll_dice_and_move`)
- Win detection and status updates
- Thread safety with RLock

**Interview Focus:**
- **Turn management**: How to handle player rotation? (List with index, modulo for wrapping)
- **State transitions**: When to change player status? (WAITING → PLAYING → WON)
- **Concurrency**: Why RLock? (Multiple threads accessing game state, reentrant for recursive calls)
- **Move validation**: Prevent cheating (only current player can roll, can't move after winning)

```python
from threading import RLock
from typing import List, Optional

class SnakeAndLadderGame:
    """Main game controller"""
    
    def __init__(self, board_size: int = 100):
        self.board = Board(board_size)
        self._players: List[Player] = []
        self._current_player_index: int = 0
        self._move_history: List[GameMove] = []
        self._dice_strategy: DiceStrategy = StandardDice()
        self._game_started = False
        self._lock = RLock()  # Thread safety for multiplayer
    
    def add_player(self, player_id: str, name: str) -> None:
        """Add player before game starts"""
        with self._lock:
            if self._game_started:
                raise RuntimeError("Cannot add players after game started")
            
            # Check for duplicate IDs
            if any(p.id == player_id for p in self._players):
                raise ValueError(f"Player {player_id} already exists")
            
            player = Player(id=player_id, name=name)
            self._players.append(player)
    
    def start_game(self) -> None:
        """Initialize game state"""
        with self._lock:
            if len(self._players) < 2:
                raise RuntimeError("Need at least 2 players")
            
            # Set all players to PLAYING
            for player in self._players:
                player.status = PlayerStatus.PLAYING
            
            self._game_started = True
            self._current_player_index = 0
    
    def get_current_player(self) -> Player:
        """Get player whose turn it is"""
        return self._players[self._current_player_index]
    
    def roll_dice_and_move(self) -> GameMove:
        """
        Core game loop: roll dice, move player, update state
        
        Interview Focus: Walk through complete flow
        1. Validate game state
        2. Get current player
        3. Roll dice
        4. Calculate next position (with snakes/ladders)
        5. Update player state
        6. Check win condition
        7. Rotate turn
        8. Return move for audit
        """
        with self._lock:
            if not self._game_started:
                raise RuntimeError("Game not started")
            
            current_player = self.get_current_player()
            
            # Can't move if already won
            if current_player.status == PlayerStatus.WON:
                raise RuntimeError(f"{current_player.name} already won")
            
            # Roll dice
            dice_roll = self._dice_strategy.roll()
            from_pos = current_player.position
            
            # Calculate next position
            to_pos, cell_type = self.board.get_next_position(from_pos, dice_roll)
            
            # Update player state
            current_player.position = to_pos
            current_player.moves_count += 1
            
            if cell_type == CellType.SNAKE:
                current_player.snakes_hit += 1
            elif cell_type == CellType.LADDER:
                current_player.ladders_climbed += 1
            
            # Check win condition
            if self.board.is_winning_position(to_pos):
                current_player.status = PlayerStatus.WON
            
            # Record move
            move = GameMove(
                player_id=current_player.id,
                dice_roll=dice_roll,
                from_position=from_pos,
                to_position=to_pos,
                cell_type=cell_type
            )
            self._move_history.append(move)
            
            # Rotate turn (only if not won)
            if current_player.status != PlayerStatus.WON:
                self._advance_turn()
            
            return move
    
    def _advance_turn(self) -> None:
        """Move to next player (skip winners)"""
        active_players = [p for p in self._players if p.status == PlayerStatus.PLAYING]
        
        if not active_players:
            return  # Game over
        
        # Circular rotation with modulo
        self._current_player_index = (self._current_player_index + 1) % len(self._players)
        
        # Skip winners
        while self.get_current_player().status == PlayerStatus.WON:
            self._current_player_index = (self._current_player_index + 1) % len(self._players)
    
    def get_winner(self) -> Optional[Player]:
        """Get first player who reached final position"""
        winners = [p for p in self._players if p.status == PlayerStatus.WON]
        return winners[0] if winners else None
    
    def get_move_history(self) -> List[GameMove]:
        """Complete audit trail"""
        return self._move_history.copy()
    
    def get_player_standings(self) -> List[Player]:
        """Leaderboard sorted by position (descending)"""
        return sorted(self._players, key=lambda p: p.position, reverse=True)
```

**Key Methods:**
- `roll_dice_and_move()`: Single atomic operation (not split into roll/move)
- `_advance_turn()`: Circular rotation skipping winners
- `get_player_standings()`: Snapshot for UI updates

---

### Phase 5: Advanced Features - Statistics & Replay (25 minutes)

**What to build:**
- Game state snapshots
- Move history analysis
- Multi-game statistics
- Board visualization

**Interview Focus:**
- **Audit trail**: Why store complete move history? (Replay, cheat detection, analytics)
- **Aggregation**: How to analyze 1000s of games? (Average moves, win rate by position)
- **Memory management**: What if move history grows too large? (Pagination, streaming to disk)

```python
from typing import Dict, Any

class SnakeAndLadderGame:
    # ... (previous methods)
    
    def get_game_state(self) -> Dict[str, Any]:
        """Complete game snapshot for persistence/UI"""
        with self._lock:
            return {
                "board_size": self.board.size,
                "num_snakes": len(self.board._snakes),
                "num_ladders": len(self.board._ladders),
                "players": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "position": p.position,
                        "status": p.status.value,
                        "moves": p.moves_count,
                        "snakes_hit": p.snakes_hit,
                        "ladders_climbed": p.ladders_climbed
                    }
                    for p in self._players
                ],
                "current_turn": self.get_current_player().name if self._game_started else None,
                "total_moves": len(self._move_history),
                "winner": self.get_winner().name if self.get_winner() else None
            }
    
    def get_board_visualization(self) -> str:
        """Text representation of board state"""
        lines = [
            f"{'='*60}",
            f"BOARD (Size: {self.board.size})",
            f"{'='*60}",
            "",
            f"🐍 Snakes ({len(self.board._snakes)}):"
        ]
        
        for pos, snake in sorted(self.board._snakes.items()):
            lines.append(f"  Snake({snake.head}→{snake.tail})")
        
        lines.append("")
        lines.append(f"🪜 Ladders ({len(self.board._ladders)}):")
        for pos, ladder in sorted(self.board._ladders.items()):
            lines.append(f"  Ladder({ladder.bottom}→{ladder.top})")
        
        lines.append("")
        lines.append(f"👥 Players ({len(self._players)}):")
        for p in self._players:
            status_emoji = {"waiting": "⏸️", "playing": "▶️", "won": "🏆"}[p.status.value]
            lines.append(f"  {status_emoji} {p.name}: Position {p.position}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def play_full_game(self, max_moves: int = 1000) -> Player:
        """
        Automated gameplay until win (for testing)
        
        Interview Focus: When would you use this?
        - Load testing (simulate 10,000 games)
        - AI training (generate training data)
        - Performance profiling
        """
        for _ in range(max_moves):
            self.roll_dice_and_move()
            
            winner = self.get_winner()
            if winner:
                return winner
        
        raise RuntimeError(f"No winner after {max_moves} moves")

# Multi-game statistics
def analyze_multiple_games(num_games: int = 100) -> Dict[str, float]:
    """Run simulations and calculate averages"""
    total_moves = []
    player1_wins = 0
    
    for _ in range(num_games):
        game = SnakeAndLadderGame(board_size=100)
        game.setup_classic_board()
        game.add_player("p1", "Player 1")
        game.add_player("p2", "Player 2")
        game.start_game()
        
        winner = game.play_full_game()
        total_moves.append(len(game.get_move_history()))
        
        if winner.id == "p1":
            player1_wins += 1
    
    return {
        "avg_moves": sum(total_moves) / len(total_moves),
        "min_moves": min(total_moves),
        "max_moves": max(total_moves),
        "player1_win_rate": player1_wins / num_games
    }
```

---

### Phase 6: Classic Board Setup & Demo (20 minutes)

**What to build:**
- `setup_classic_board()` with traditional snake/ladder placements
- Demo functions for different scenarios
- Testing harness

```python
class SnakeAndLadderGame:
    # ... (previous methods)
    
    def setup_classic_board(self) -> None:
        """
        Traditional 100-cell board configuration
        
        Interview Focus: How were these positions chosen?
        - Historical game design (balance luck vs skill)
        - Snakes near end (increase tension)
        - Ladders early (speed up start)
        """
        # Classic snakes (10)
        snakes = [
            (99, 54), (97, 75), (95, 72), (93, 73), (87, 24),
            (70, 55), (64, 36), (62, 18), (52, 42), (25, 2)
        ]
        
        for head, tail in snakes:
            self.board.add_snake(head, tail)
        
        # Classic ladders (10)
        ladders = [
            (3, 51), (6, 27), (20, 70), (36, 55), (43, 77),
            (47, 53), (60, 72), (68, 98), (74, 92), (80, 94)
        ]
        
        for bottom, top in ladders:
            self.board.add_ladder(bottom, top)

# Demo function
def demo_complete_game():
    """Full game with weighted dice"""
    print("="*70)
    print("  Complete Game Simulation")
    print("="*70)
    
    # Create game with biased dice (faster gameplay)
    game = SnakeAndLadderGame(board_size=100)
    game.setup_classic_board()
    game.set_dice_strategy(WeightedDice([0.05, 0.05, 0.1, 0.15, 0.3, 0.35]))
    
    game.add_player("p1", "Player 1")
    game.add_player("p2", "Player 2")
    game.start_game()
    
    print("\n🔹 Starting game with weighted dice (favors 5-6)...")
    
    # Play until someone wins
    winner = game.play_full_game()
    
    print(f"  {winner.name} won in {len(game.get_move_history())} moves!")
    
    # Show final standings
    print("\n🔹 Final standings:")
    for player in game.get_player_standings():
        status = "WON" if player.status == PlayerStatus.WON else "PLAYING"
        print(f"  {player.name}: Position {player.position} "
              f"({player.moves_count} moves, {player.snakes_hit} snakes, "
              f"{player.ladders_climbed} ladders) - {status}")
    
    # Show last 5 moves
    print("\n🔹 Last 5 moves:")
    for move in game.get_move_history()[-5:]:
        event = {"normal": "normal", "snake": "snake", "ladder": "ladder"}.get(
            move.cell_type.value, "boundary"
        )
        player_name = next(p.name for p in game._players if p.id == move.player_id)
        print(f"  Move #{len(game.get_move_history()) - game.get_move_history().index(move)}: "
              f"{player_name} rolled {move.dice_roll}, "
              f"{move.from_position}→{move.to_position} [{event}]")
```

---

## Critical Knowledge Points

### 1. Strategy Pattern for Dice Rolling

**Why it matters:**
- **Testability**: Inject ControlledDice for deterministic tests
- **Game modes**: Easy to add loaded dice, custom probabilities
- **Open/Closed**: Add strategies without modifying SnakeAndLadderGame

**When to mention:**
- Discussing design patterns in interviews
- Explaining how to write testable code
- Showing understanding of dependency injection

### 2. Dictionary-Based Board Representation

**Complexity:**
- Lookup: O(1)
- Insertion: O(1)
- Space: O(S + L) where S = snakes, L = ladders

**Alternative (2D array):**
- Space: O(N) for N cells (wasteful for sparse data)
- Position-to-coordinates conversion: O(1) but complex

**Trade-offs:**
- Dict is better for sparse data (10-20 snakes/ladders on 100 cells)
- Array is better for dense data or grid-based operations

### 3. Thread Safety with RLock

**Why RLock?**
- **Reentrant**: Same thread can acquire lock multiple times
- **Multiplayer**: Prevents race conditions when multiple threads access game state

**Example race condition:**
```python
# Without lock:
def roll_dice_and_move():
    player = self.get_current_player()  # Thread A reads index
    # [Thread B changes _current_player_index]
    dice = self.roll()  # Thread A uses stale player reference
    # Result: Wrong player moves!
```

### 4. Move History as Audit Trail

**Use cases:**
- **Replay**: Reconstruct game state at any point
- **Cheat detection**: Verify moves follow rules
- **Analytics**: Average moves to win, snake hit frequency
- **ML training**: Generate labeled data for AI

**Optimization for large games:**
```python
# Paginated history
def get_move_history_page(self, page: int, size: int = 100) -> List[GameMove]:
    start = page * size
    return self._move_history[start:start + size]
```

### 5. Boundary Handling for Winning

**Rule**: Must roll exact number to reach 100

**Implementation:**
```python
if target > self.size:
    return current, CellType.NORMAL  # Stay at current position
```

**Why this rule?**
- Increases game duration (more tension)
- Prevents lucky wins with 6 at position 95

---

## Expected Interview Questions

### Q1: Why use Strategy pattern for dice instead of just `random.randint(1, 6)`?

**Answer:**

The Strategy pattern provides critical benefits:

1. **Testability**: Can inject ControlledDice for deterministic unit tests
2. **Extensibility**: Easy to add new dice types (loaded, N-sided, double dice)
3. **Open/Closed Principle**: Add strategies without modifying game logic
4. **Flexibility**: Switch dice at runtime for different game modes

**Code Example:**
```python
# Without Strategy: Hard to test
class SnakeAndLadderGame:
    def roll_dice(self):
        return random.randint(1, 6)  # Can't control for testing

# With Strategy: Easy to test
def test_exact_win():
    game = SnakeAndLadderGame()
    game.set_dice_strategy(ControlledDice([6, 6, 6, 4]))  # Predetermined
    # Now we can test exact scenarios
```

**When to use:**
- When you need multiple algorithms for the same operation
- When algorithm selection should be flexible at runtime
- When you want to decouple algorithm from client code

---

### Q2: Why use dictionaries for board instead of 2D array? What are the trade-offs?

**Answer:**

**Dictionary Approach** (chosen):
```python
_position_map: Dict[int, int] = {  # Only special cells
    17: 7,   # Snake
    28: 84,  # Ladder
}
```

**Advantages:**
- O(1) lookup for snakes/ladders
- Memory efficient: O(S + L) vs O(N) for array
- Sparse data: 10-20 special cells on 100-cell board
- Easy to add/remove snakes/ladders dynamically

**2D Array Approach:**
```python
board = [[None] * 10 for _ in range(10)]  # 100 cells
board[1][7] = Snake(17, 7)  # Complex position math
```

**Disadvantages of array:**
- Wastes memory (80+ empty cells)
- Position-to-coordinates conversion: `row = pos // 10, col = pos % 10`
- Better only if dense data (every cell has state)

**Trade-off Decision Matrix:**
| Factor | Dict | Array |
|--------|------|-------|
| Lookup | O(1) | O(1) |
| Space | O(S+L) | O(N) |
| Sparse data | ✅ Better | ❌ Wasteful |
| Dense data | ❌ | ✅ Better |
| Add/remove | ✅ Easy | ❌ Complex |

**Verdict**: For Snake & Ladder (sparse board), dictionaries are optimal.

---

### Q3: How would you test the game logic with controlled dice rolls?

**Answer:**

Use ControlledDice with predetermined rolls:

```python
from collections import deque

class ControlledDice(DiceStrategy):
    """Inject predetermined rolls for testing"""
    def __init__(self, rolls: List[int]):
        self.rolls = deque(rolls)
    
    def roll(self) -> int:
        return self.rolls.popleft()

# Test: Player wins in 2 moves
def test_quick_win():
    game = SnakeAndLadderGame(board_size=20)
    game.board.add_ladder(18, 20)  # Ladder near end
    
    # Predetermined: 18 (land on ladder) → win
    game.set_dice_strategy(ControlledDice([18]))
    game.add_player("test", "Tester")
    game.start_game()
    
    move = game.roll_dice_and_move()
    assert move.to_position == 20  # Won via ladder
    assert game.get_winner().id == "test"

# Test: Boundary overflow (stays at current)
def test_boundary_overflow():
    game = SnakeAndLadderGame(board_size=100)
    game.set_dice_strategy(ControlledDice([6]))  # Roll 6
    game.add_player("test", "Tester")
    game.start_game()
    
    # Manually set position to 96
    game._players[0].position = 96
    
    move = game.roll_dice_and_move()
    # 96 + 6 = 102 > 100, so stays at 96
    assert move.to_position == 96

# Test: Snake hits counted correctly
def test_snake_statistics():
    game = SnakeAndLadderGame(board_size=100)
    game.board.add_snake(17, 7)
    game.set_dice_strategy(ControlledDice([17]))  # Land on snake
    game.add_player("test", "Tester")
    game.start_game()
    
    game.roll_dice_and_move()
    player = game._players[0]
    assert player.snakes_hit == 1
    assert player.position == 7
```

**Key Testing Strategies:**
1. **Controlled rolls**: Inject exact dice values
2. **Edge cases**: Boundary overflow, exact win, first/last position
3. **Statistics**: Verify counters (snakes_hit, ladders_climbed, moves_count)
4. **Turn rotation**: Test player order with 3+ players

---

### Q4: How would you handle multiplayer concurrency? What if two players try to roll simultaneously?

**Answer:**

**Use threading.RLock for reentrant locking:**

```python
from threading import RLock

class SnakeAndLadderGame:
    def __init__(self):
        self._lock = RLock()  # Reentrant lock
        self._current_player_index = 0
    
    def roll_dice_and_move(self) -> GameMove:
        with self._lock:  # Atomic operation
            # 1. Validate current player
            current = self.get_current_player()
            if current.status == PlayerStatus.WON:
                raise RuntimeError("Player already won")
            
            # 2. Roll dice
            dice = self._dice_strategy.roll()
            
            # 3. Update position
            new_pos, cell_type = self.board.get_next_position(current.position, dice)
            current.position = new_pos
            
            # 4. Rotate turn
            self._advance_turn()
            
            return GameMove(...)
```

**Why RLock?**
- **Reentrant**: Same thread can acquire multiple times (for nested calls)
- **Thread-safe**: Prevents race conditions:

```python
# Race condition without lock:
# Thread A: player = get_current_player()  # Gets Player 1
# [Thread B changes _current_player_index to 2]
# Thread A: player.position += dice  # Updates wrong player!
```

**Alternative: Queue-based turns**
```python
from queue import Queue

class SnakeAndLadderGame:
    def __init__(self):
        self._turn_queue = Queue()  # Thread-safe
    
    def roll_dice_and_move(self):
        player_id = self._turn_queue.get()  # Blocks until turn
        # ... move logic
        self._turn_queue.put(player_id)  # Re-enqueue if not won
```

**Trade-offs:**
| Approach | Pros | Cons |
|----------|------|------|
| RLock | Simple, fast | Requires careful lock scope |
| Queue | Natural turn ordering | More complex setup |
| Actor model | Scalable | Overkill for simple game |

**Production considerations:**
- **Timeouts**: Player has 30 seconds to roll, else auto-skip
- **Disconnections**: Mark player as inactive, continue game
- **Replay**: Persist move history to database for crash recovery

---

### Q5: How would you extend this to support power-ups (e.g., extra dice roll, skip snake)?

**Answer:**

**Design Pattern: Decorator + Command**

```python
from abc import ABC, abstractmethod
from enum import Enum

class PowerUpType(Enum):
    EXTRA_ROLL = "extra_roll"
    SKIP_SNAKE = "skip_snake"
    TELEPORT = "teleport"

@dataclass
class PowerUp:
    """Power-up inventory item"""
    type: PowerUpType
    uses_remaining: int
    description: str

class MoveBehavior(ABC):
    """Decorator for move behavior"""
    @abstractmethod
    def execute(self, player: Player, dice_roll: int, board: Board) -> int:
        pass

class NormalMove(MoveBehavior):
    """Standard move logic"""
    def execute(self, player: Player, dice_roll: int, board: Board) -> int:
        return board.get_next_position(player.position, dice_roll)[0]

class SkipSnakeMove(MoveBehavior):
    """Ignores snakes for this move"""
    def execute(self, player: Player, dice_roll: int, board: Board) -> int:
        target = player.position + dice_roll
        # Check if snake, but don't apply it
        if target in board._snakes:
            return target  # Stay at snake head (don't slide down)
        return board.get_next_position(player.position, dice_roll)[0]

# Extended Player class
@dataclass
class Player:
    # ... existing fields
    power_ups: Dict[PowerUpType, PowerUp] = field(default_factory=dict)
    
    def add_power_up(self, power_up: PowerUp):
        self.power_ups[power_up.type] = power_up
    
    def use_power_up(self, power_up_type: PowerUpType) -> bool:
        if power_up_type not in self.power_ups:
            return False
        
        power_up = self.power_ups[power_up_type]
        if power_up.uses_remaining <= 0:
            return False
        
        power_up.uses_remaining -= 1
        if power_up.uses_remaining == 0:
            del self.power_ups[power_up_type]
        
        return True

# Extended game logic
class SnakeAndLadderGame:
    def roll_dice_and_move(self, use_power_up: Optional[PowerUpType] = None) -> GameMove:
        with self._lock:
            current = self.get_current_player()
            
            # Activate power-up
            move_behavior: MoveBehavior = NormalMove()
            if use_power_up == PowerUpType.SKIP_SNAKE:
                if current.use_power_up(PowerUpType.SKIP_SNAKE):
                    move_behavior = SkipSnakeMove()
            
            dice = self._dice_strategy.roll()
            
            # Apply power-up: extra roll
            if use_power_up == PowerUpType.EXTRA_ROLL:
                if current.use_power_up(PowerUpType.EXTRA_ROLL):
                    dice += self._dice_strategy.roll()
            
            # Execute move with behavior
            new_pos = move_behavior.execute(current, dice, self.board)
            current.position = new_pos
            
            # ... rest of logic
```

**Power-up distribution strategies:**
1. **Random cells**: Land on cell 42 → get power-up
2. **Shop**: Spend coins earned from moves
3. **Achievements**: Hit 5 ladders → unlock power-up

**Testing power-ups:**
```python
def test_skip_snake_power_up():
    game = SnakeAndLadderGame()
    game.board.add_snake(17, 7)
    game.set_dice_strategy(ControlledDice([17]))
    
    player = game.add_player("test", "Tester")
    player.add_power_up(PowerUp(PowerUpType.SKIP_SNAKE, uses_remaining=1, description="Skip snake"))
    
    game.start_game()
    move = game.roll_dice_and_move(use_power_up=PowerUpType.SKIP_SNAKE)
    
    # Without power-up: 17 → 7 (snake)
    # With power-up: 17 → 17 (ignore snake)
    assert move.to_position == 17
    assert player.power_ups == {}  # Used up
```

---

### Q6: How would you implement game state persistence for save/load functionality?

**Answer:**

**Approach: JSON serialization with versioning**

```python
import json
from typing import Dict, Any
from datetime import datetime

class SnakeAndLadderGame:
    
    def serialize_game_state(self) -> str:
        """Convert game to JSON for persistence"""
        state = {
            "version": "1.0",  # For backward compatibility
            "timestamp": datetime.now().isoformat(),
            "board": {
                "size": self.board.size,
                "snakes": [
                    {"head": s.head, "tail": s.tail}
                    for s in self.board._snakes.values()
                ],
                "ladders": [
                    {"bottom": l.bottom, "top": l.top}
                    for l in self.board._ladders.values()
                ]
            },
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "position": p.position,
                    "status": p.status.value,
                    "moves_count": p.moves_count,
                    "snakes_hit": p.snakes_hit,
                    "ladders_climbed": p.ladders_climbed
                }
                for p in self._players
            ],
            "current_player_index": self._current_player_index,
            "game_started": self._game_started,
            "move_history": [
                {
                    "player_id": m.player_id,
                    "dice_roll": m.dice_roll,
                    "from_position": m.from_position,
                    "to_position": m.to_position,
                    "cell_type": m.cell_type.value,
                    "timestamp": m.timestamp
                }
                for m in self._move_history
            ]
        }
        return json.dumps(state, indent=2)
    
    @staticmethod
    def deserialize_game_state(json_str: str) -> 'SnakeAndLadderGame':
        """Restore game from JSON"""
        state = json.loads(json_str)
        
        # Validate version
        if state["version"] != "1.0":
            raise ValueError(f"Unsupported version: {state['version']}")
        
        # Reconstruct board
        game = SnakeAndLadderGame(board_size=state["board"]["size"])
        
        for snake in state["board"]["snakes"]:
            game.board.add_snake(snake["head"], snake["tail"])
        
        for ladder in state["board"]["ladders"]:
            game.board.add_ladder(ladder["bottom"], ladder["top"])
        
        # Reconstruct players
        for player_data in state["players"]:
            player = Player(
                id=player_data["id"],
                name=player_data["name"],
                position=player_data["position"],
                status=PlayerStatus(player_data["status"]),
                moves_count=player_data["moves_count"],
                snakes_hit=player_data["snakes_hit"],
                ladders_climbed=player_data["ladders_climbed"]
            )
            game._players.append(player)
        
        # Restore game state
        game._current_player_index = state["current_player_index"]
        game._game_started = state["game_started"]
        
        # Restore move history
        for move_data in state["move_history"]:
            move = GameMove(
                player_id=move_data["player_id"],
                dice_roll=move_data["dice_roll"],
                from_position=move_data["from_position"],
                to_position=move_data["to_position"],
                cell_type=CellType(move_data["cell_type"]),
                timestamp=move_data["timestamp"]
            )
            game._move_history.append(move)
        
        return game
    
    def save_to_file(self, filepath: str):
        """Persist game to disk"""
        with open(filepath, 'w') as f:
            f.write(self.serialize_game_state())
    
    @staticmethod
    def load_from_file(filepath: str) -> 'SnakeAndLadderGame':
        """Load game from disk"""
        with open(filepath, 'r') as f:
            return SnakeAndLadderGame.deserialize_game_state(f.read())

# Usage example
game = SnakeAndLadderGame()
game.setup_classic_board()
game.add_player("p1", "Alice")
game.add_player("p2", "Bob")
game.start_game()

# Play 10 moves
for _ in range(10):
    game.roll_dice_and_move()

# Save game
game.save_to_file("saved_game.json")

# Later: restore game
restored = SnakeAndLadderGame.load_from_file("saved_game.json")
restored.roll_dice_and_move()  # Continue from where you left off
```

**Production considerations:**
- **Compression**: Use gzip for large move histories
- **Database**: Store in PostgreSQL/MongoDB for multi-user games
- **Incremental saves**: Only save deltas (last N moves) to reduce I/O
- **Cloud storage**: S3/Azure Blob for scalability

---

### Q7: What optimizations would you make for very large boards (10,000+ cells)?

**Answer:**

**1. Sparse Data Structures (already optimal)**
```python
# Current: O(S + L) space for S snakes, L ladders
_position_map: Dict[int, int]  # Only special cells

# Alternative if dense board (many snakes/ladders):
# Use compressed bitmap or segment tree
```

**2. Lazy Board Generation**
```python
class LazyBoard:
    """Generate snakes/ladders on-demand"""
    def __init__(self, size: int, snake_density: float = 0.1):
        self.size = size
        self.snake_density = snake_density
        self._cache: Dict[int, int] = {}  # Memoize generated snakes
    
    def get_next_position(self, pos: int, dice: int) -> int:
        target = pos + dice
        
        # Check cache first
        if target in self._cache:
            return self._cache[target]
        
        # Generate snake/ladder probabilistically
        if random.random() < self.snake_density:
            # Generate snake: go back 10-30%
            new_pos = target - int(target * random.uniform(0.1, 0.3))
            self._cache[target] = new_pos
            return new_pos
        
        return target
```

**3. Move History Pagination**
```python
# Instead of storing all moves in memory:
class SnakeAndLadderGame:
    def __init__(self):
        self._move_history_file = open("moves.log", "a")
    
    def roll_dice_and_move(self):
        # ... move logic
        
        # Stream to disk instead of memory
        self._move_history_file.write(json.dumps(move_dict) + "\n")
        self._move_history_file.flush()
    
    def get_move_history_page(self, page: int, size: int = 100):
        """Read paginated history from disk"""
        with open("moves.log", "r") as f:
            moves = [json.loads(line) for line in f]
            start = page * size
            return moves[start:start + size]
```

**4. Position Encoding for Memory Efficiency**
```python
# Instead of storing full GameMove objects:
# Encode move as packed bytes (8 bytes instead of 40+)

import struct

def encode_move(player_id: int, dice: int, from_pos: int, to_pos: int) -> bytes:
    """Pack into 8 bytes: [player:2][dice:1][from:2][to:2][type:1]"""
    return struct.pack("HBHHB", player_id, dice, from_pos, to_pos, cell_type)

def decode_move(data: bytes) -> tuple:
    return struct.pack("HBHHB", data)
```

**5. Parallel Move Validation (for AI simulations)**
```python
from multiprocessing import Pool

def simulate_game(game_config: dict) -> dict:
    """Worker process: run one game"""
    game = SnakeAndLadderGame(**game_config)
    winner = game.play_full_game()
    return {"winner": winner.id, "moves": len(game.get_move_history())}

# Run 10,000 games in parallel
with Pool(processes=8) as pool:
    results = pool.map(simulate_game, [{"board_size": 10000}] * 10000)
    avg_moves = sum(r["moves"] for r in results) / len(results)
```

**Complexity Comparison:**

| Operation | Current | Optimized (10k cells) |
|-----------|---------|----------------------|
| Add snake | O(1) | O(1) |
| Position lookup | O(1) | O(1) cached, O(log N) uncached |
| Move history | O(N) space | O(1) space (disk streaming) |
| 1000 games | Serial | Parallel (8x faster) |

**When to optimize:**
- Board size > 1000 cells
- Move history > 100k moves
- Simulating millions of games (ML training)

---

## Testing Strategy

### Unit Tests

```python
import unittest

class TestSnakeAndLadderGame(unittest.TestCase):
    
    def test_add_invalid_snake(self):
        """Snake tail must be < head"""
        board = Board(size=100)
        with self.assertRaises(ValueError):
            board.add_snake(head=10, tail=20)  # Invalid
    
    def test_overlap_prevention(self):
        """Can't add snake and ladder at same position"""
        board = Board(size=100)
        board.add_snake(17, 7)
        with self.assertRaises(ValueError):
            board.add_ladder(17, 28)  # Overlap!
    
    def test_boundary_overflow(self):
        """Rolling past 100 keeps you at current position"""
        game = SnakeAndLadderGame(board_size=100)
        game.set_dice_strategy(ControlledDice([6]))
        game.add_player("test", "Tester")
        game.start_game()
        
        game._players[0].position = 98  # Manually set
        move = game.roll_dice_and_move()
        
        # 98 + 6 = 104 > 100, so stays at 98
        assert move.to_position == 98
    
    def test_exact_win(self):
        """Must roll exact number to win"""
        game = SnakeAndLadderGame(board_size=100)
        game.set_dice_strategy(ControlledDice([2]))
        game.add_player("test", "Tester")
        game.start_game()
        
        game._players[0].position = 98
        move = game.roll_dice_and_move()
        
        assert move.to_position == 100
        assert game.get_winner().id == "test"
    
    def test_turn_rotation(self):
        """Players take turns in order"""
        game = SnakeAndLadderGame()
        game.add_player("p1", "Alice")
        game.add_player("p2", "Bob")
        game.start_game()
        
        assert game.get_current_player().id == "p1"
        game.roll_dice_and_move()
        assert game.get_current_player().id == "p2"
        game.roll_dice_and_move()
        assert game.get_current_player().id == "p1"  # Wrapped
    
    def test_snake_statistics(self):
        """Hitting snake increments counter"""
        game = SnakeAndLadderGame()
        game.board.add_snake(17, 7)
        game.set_dice_strategy(ControlledDice([17]))
        game.add_player("test", "Tester")
        game.start_game()
        
        game.roll_dice_and_move()
        assert game._players[0].snakes_hit == 1
        assert game._players[0].position == 7
```

### Integration Tests

```python
def test_full_game_simulation():
    """Complete game from start to finish"""
    game = SnakeAndLadderGame(board_size=100)
    game.setup_classic_board()
    game.add_player("p1", "Player 1")
    game.add_player("p2", "Player 2")
    game.start_game()
    
    winner = game.play_full_game(max_moves=1000)
    
    assert winner is not None
    assert winner.position == 100
    assert len(game.get_move_history()) > 0
    assert winner.moves_count == len([m for m in game.get_move_history() if m.player_id == winner.id])

def test_save_load_persistence():
    """Game state survives serialization"""
    game = SnakeAndLadderGame()
    game.setup_classic_board()
    game.add_player("p1", "Alice")
    game.start_game()
    
    for _ in range(10):
        game.roll_dice_and_move()
    
    # Save and load
    json_state = game.serialize_game_state()
    restored = SnakeAndLadderGame.deserialize_game_state(json_state)
    
    assert restored.board.size == game.board.size
    assert len(restored._players) == len(game._players)
    assert restored._current_player_index == game._current_player_index
    assert len(restored.get_move_history()) == len(game.get_move_history())
```

---

## Production Considerations

### 1. Scalability
- **Distributed games**: Use Redis for shared game state across servers
- **Leaderboards**: PostgreSQL with indexed player_id + game_id
- **Real-time multiplayer**: WebSocket for live updates

### 2. Monitoring
- **Metrics**: Average game duration, moves per win, snake/ladder hit rates
- **Logging**: Audit trail of all moves for dispute resolution
- **Alerts**: Detect stuck games (>1000 moves without winner)

### 3. Security
- **Cheating prevention**: Validate dice rolls server-side
- **Rate limiting**: Max 1 roll per second per player
- **Input validation**: Sanitize player names, board configs

### 4. Performance
- **Caching**: Redis cache for active games (in-memory state)
- **Database indexing**: Index on (game_id, player_id, timestamp)
- **Connection pooling**: Reuse DB connections for high throughput

### 5. User Experience
- **Animations**: Smooth piece movement on UI
- **Sound effects**: Snake hiss, ladder climb sounds
- **Replay mode**: Watch game history with playback controls
- **Tournament mode**: Bracket-based elimination

---

## Summary

### Do's ✅
- Use Strategy pattern for flexible dice rolling
- Use dictionaries for sparse board data (O(1) lookups)
- Implement thread safety with RLock for multiplayer
- Store complete move history for audit/replay
- Validate all inputs (snake/ladder positions, player IDs)
- Write unit tests with ControlledDice
- Separate data models from business logic

### Don'ts ❌
- Don't use 2D arrays for sparse boards (memory waste)
- Don't skip boundary checks (exact win rule)
- Don't allow overlapping snakes/ladders
- Don't forget turn rotation after each move
- Don't ignore thread safety in multiplayer
- Don't store move history in memory for very long games

### Key Takeaways
1. **Strategy pattern** enables testable and extensible dice logic
2. **Dictionary-based board** optimal for sparse data (10-20 items on 100 cells)
3. **Thread safety** critical for concurrent player moves
4. **Move history** provides complete audit trail for replay/analysis
5. **Boundary handling** prevents invalid wins (must roll exact number)

### Complexity Summary
| Operation | Time | Space |
|-----------|------|-------|
| Add snake/ladder | O(1) | O(S + L) |
| Position lookup | O(1) | - |
| Roll and move | O(1) | O(1) |
| Get standings | O(P log P) | O(P) |
| Move history | O(M) | O(M) |

Where: S = snakes, L = ladders, P = players, M = moves

---

## Design Patterns Used

1. **Strategy**: DiceStrategy for flexible rolling algorithms
2. **Singleton**: Single game instance per session
3. **Observer**: Future extension for game events (player moved, snake hit)
4. **Command**: GameMove as command object for undo/redo
5. **Facade**: SnakeAndLadderGame hides complexity of Board/Player/Dice

---

*This implementation demonstrates interview-level understanding of design patterns, data structures, thread safety, and production-grade software engineering practices.*
