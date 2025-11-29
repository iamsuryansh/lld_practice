"""
Snake and Ladder Game - Single File Implementation
For coding interviews and production-ready reference

Features:
- Classic board game with snakes and ladders
- Multiple players with turn-based gameplay
- Dice rolling with configurable strategy
- Win condition detection
- Power-ups and special cells (extensible)
- Game state tracking and history

Interview Focus:
- Strategy pattern for dice rolling
- Singleton pattern for game instance
- Composite pattern for board cells
- Observer pattern for game events
- Thread safety for multiplayer
- Clean OOP design with SOLID principles

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from threading import RLock
import random
import time
import uuid


# ============================================================================
# SECTION 1: MODELS - Core data classes and enums
# ============================================================================

class CellType(Enum):
    """Types of cells on the board"""
    NORMAL = "normal"
    SNAKE_HEAD = "snake_head"
    SNAKE_TAIL = "snake_tail"
    LADDER_BOTTOM = "ladder_bottom"
    LADDER_TOP = "ladder_top"
    POWER_UP = "power_up"


class PlayerStatus(Enum):
    """Player game status"""
    WAITING = "waiting"
    PLAYING = "playing"
    WON = "won"


@dataclass
class Snake:
    """
    Snake on the board - moves player down
    
    Interview Focus: Why separate class? Encapsulation and validation
    """
    head: int  # Higher position
    tail: int  # Lower position
    
    def __post_init__(self):
        if self.head <= self.tail:
            raise ValueError(f"Snake head ({self.head}) must be > tail ({self.tail})")
    
    def __str__(self):
        return f"Snake({self.head}→{self.tail})"


@dataclass
class Ladder:
    """
    Ladder on the board - moves player up
    
    Interview Focus: Similar structure to Snake but semantically different
    """
    bottom: int  # Lower position
    top: int     # Higher position
    
    def __post_init__(self):
        if self.bottom >= self.top:
            raise ValueError(f"Ladder bottom ({self.bottom}) must be < top ({self.top})")
    
    def __str__(self):
        return f"Ladder({self.bottom}→{self.top})"


@dataclass
class Player:
    """
    Player in the game
    
    Interview Focus: Track state, position, and history
    """
    player_id: str
    name: str
    position: int = 0  # Start at position 0 (before board)
    status: PlayerStatus = PlayerStatus.WAITING
    moves_count: int = 0
    snakes_hit: int = 0
    ladders_climbed: int = 0
    
    def move_to(self, new_position: int):
        """Move player to new position"""
        self.position = new_position
        self.moves_count += 1
    
    def hit_snake(self):
        """Record snake hit"""
        self.snakes_hit += 1
    
    def climb_ladder(self):
        """Record ladder climb"""
        self.ladders_climbed += 1


@dataclass
class GameMove:
    """
    Record of a single move in the game
    
    Interview Focus: Audit trail for game replay and analysis
    """
    move_number: int
    player_id: str
    player_name: str
    dice_roll: int
    start_position: int
    end_position: int
    event: str  # "normal", "snake", "ladder", "won"
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        return (f"Move #{self.move_number}: {self.player_name} rolled {self.dice_roll}, "
                f"{self.start_position}→{self.end_position} [{self.event}]")


# ============================================================================
# SECTION 2: DICE STRATEGY - Strategy Pattern
# ============================================================================

class DiceStrategy(ABC):
    """
    Abstract dice rolling strategy
    
    Strategy Pattern: Different dice rolling behaviors
    
    Interview Focus: Why Strategy Pattern?
    - Easy to add new dice types (weighted, loaded, etc.)
    - Testing with controlled dice rolls
    - Different game modes (easy/hard)
    """
    
    @abstractmethod
    def roll(self) -> int:
        """Roll the dice and return result"""
        pass
    
    @abstractmethod
    def get_sides(self) -> int:
        """Get number of sides on the dice"""
        pass


class StandardDice(DiceStrategy):
    """
    Standard 6-sided fair dice
    
    Interview Focus: Random number generation with proper range
    """
    
    def __init__(self, sides: int = 6):
        self.sides = sides
    
    def roll(self) -> int:
        """
        Roll fair dice
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return random.randint(1, self.sides)
    
    def get_sides(self) -> int:
        return self.sides


class ControlledDice(DiceStrategy):
    """
    Controlled dice for testing
    
    Interview Focus: Testability - inject predetermined rolls
    """
    
    def __init__(self, predetermined_rolls: List[int]):
        self.rolls = deque(predetermined_rolls)
        self.sides = 6
    
    def roll(self) -> int:
        """Return next predetermined roll"""
        if not self.rolls:
            return random.randint(1, self.sides)
        return self.rolls.popleft()
    
    def get_sides(self) -> int:
        return self.sides


class WeightedDice(DiceStrategy):
    """
    Weighted dice - some numbers more likely
    
    Interview Focus: Probability distribution manipulation
    """
    
    def __init__(self, weights: Dict[int, float]):
        """
        Args:
            weights: {face_value: probability}
            Example: {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.2, 6: 0.3}
        """
        self.faces = list(weights.keys())
        self.weights = list(weights.values())
        self.sides = max(self.faces)
    
    def roll(self) -> int:
        """Roll with weighted probability"""
        return random.choices(self.faces, weights=self.weights, k=1)[0]
    
    def get_sides(self) -> int:
        return self.sides


# ============================================================================
# SECTION 3: BOARD - Game board with cells
# ============================================================================

class Board:
    """
    Game board with snakes and ladders
    
    Responsibilities:
    - Manage board size
    - Place snakes and ladders
    - Handle position movements
    - Validate board configuration
    
    Interview Focus: Data structure choice - dict for O(1) lookup
    """
    
    def __init__(self, size: int = 100):
        """
        Initialize board
        
        Args:
            size: Number of cells (default 100 for classic game)
        """
        self.size = size
        self.snakes: Dict[int, Snake] = {}  # head_position -> Snake
        self.ladders: Dict[int, Ladder] = {}  # bottom_position -> Ladder
        self.power_ups: Dict[int, str] = {}  # position -> power_up_type
        
        # Track all special positions for quick lookup
        self.special_positions: Set[int] = set()
    
    def add_snake(self, head: int, tail: int) -> Tuple[bool, str]:
        """
        Add snake to board
        
        Interview Focus: Validation - prevent overlaps and invalid positions
        """
        # Validate positions
        if head < 2 or head > self.size:
            return False, f"Snake head must be between 2 and {self.size}"
        
        if tail < 1 or tail >= head:
            return False, f"Snake tail must be between 1 and {head-1}"
        
        # Check for overlaps
        if head in self.special_positions:
            return False, f"Position {head} already occupied"
        
        if tail in self.special_positions:
            return False, f"Position {tail} already occupied"
        
        # Add snake
        snake = Snake(head, tail)
        self.snakes[head] = snake
        self.special_positions.add(head)
        self.special_positions.add(tail)
        
        return True, f"Added {snake}"
    
    def add_ladder(self, bottom: int, top: int) -> Tuple[bool, str]:
        """
        Add ladder to board
        
        Interview Focus: Similar validation to snake but different semantics
        """
        # Validate positions
        if bottom < 1 or bottom >= self.size:
            return False, f"Ladder bottom must be between 1 and {self.size-1}"
        
        if top <= bottom or top > self.size:
            return False, f"Ladder top must be between {bottom+1} and {self.size}"
        
        # Check for overlaps
        if bottom in self.special_positions:
            return False, f"Position {bottom} already occupied"
        
        if top in self.special_positions:
            return False, f"Position {top} already occupied"
        
        # Add ladder
        ladder = Ladder(bottom, top)
        self.ladders[bottom] = ladder
        self.special_positions.add(bottom)
        self.special_positions.add(top)
        
        return True, f"Added {ladder}"
    
    def get_next_position(self, current_position: int, dice_roll: int) -> Tuple[int, str]:
        """
        Calculate next position after dice roll
        
        Interview Focus: Game logic - handle snakes, ladders, and boundaries
        
        Returns:
            (final_position, event_type)
            event_type: "normal", "snake", "ladder", "boundary"
        
        Time Complexity: O(1) with dict lookup
        """
        # Calculate new position
        new_position = current_position + dice_roll
        
        # Check boundary
        if new_position > self.size:
            return current_position, "boundary"  # Stay at current position
        
        # Check for snake
        if new_position in self.snakes:
            snake = self.snakes[new_position]
            return snake.tail, "snake"
        
        # Check for ladder
        if new_position in self.ladders:
            ladder = self.ladders[new_position]
            return ladder.top, "ladder"
        
        return new_position, "normal"
    
    def is_winning_position(self, position: int) -> bool:
        """Check if position is winning position (exactly at size)"""
        return position == self.size
    
    def get_cell_info(self, position: int) -> str:
        """Get information about a cell"""
        if position in self.snakes:
            return f"Snake head → {self.snakes[position].tail}"
        if position in self.ladders:
            return f"Ladder bottom → {self.ladders[position].top}"
        
        # Check if position is snake tail or ladder top
        for snake in self.snakes.values():
            if snake.tail == position:
                return f"Snake tail (from {snake.head})"
        
        for ladder in self.ladders.values():
            if ladder.top == position:
                return f"Ladder top (from {ladder.bottom})"
        
        return "Normal cell"


# ============================================================================
# SECTION 4: GAME ENGINE - Main game logic
# ============================================================================

class SnakeAndLadderGame:
    """
    Main game controller
    
    Responsibilities:
    - Manage players and turns
    - Execute moves with dice rolling
    - Track game state and history
    - Determine winner
    - Provide game statistics
    
    Thread Safety: Uses RLock for concurrent operations
    
    Interview Focus: How to coordinate complex game logic?
    - Clear turn management
    - Event tracking for replay
    - Extensible for power-ups and variants
    """
    
    def __init__(self, board_size: int = 100, dice_strategy: Optional[DiceStrategy] = None):
        """
        Initialize game
        
        Args:
            board_size: Size of the board (default 100)
            dice_strategy: Dice rolling strategy (default StandardDice)
        """
        self.game_id = str(uuid.uuid4())[:8]
        self.board = Board(board_size)
        self.dice = dice_strategy or StandardDice()
        
        # Players
        self.players: Dict[str, Player] = {}
        self.player_order: List[str] = []  # Track turn order
        self.current_player_index: int = 0
        
        # Game state
        self.game_started: bool = False
        self.game_ended: bool = False
        self.winner: Optional[Player] = None
        
        # History
        self.moves: List[GameMove] = []
        self.move_counter: int = 0
        
        # Thread safety
        self.lock = RLock()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    def add_player(self, name: str) -> Tuple[bool, str]:
        """
        Add player to game
        
        Interview Focus: Validation and state management
        """
        with self.lock:
            if self.game_started:
                return False, "Cannot add players after game has started"
            
            # Generate player ID
            player_id = str(uuid.uuid4())[:8]
            
            # Create player
            player = Player(player_id, name)
            self.players[player_id] = player
            self.player_order.append(player_id)
            
            return True, f"Added player {name} (ID: {player_id})"
    
    def setup_classic_board(self) -> Tuple[bool, str]:
        """
        Set up classic Snake and Ladder board
        
        Interview Focus: Standard game configuration
        Classic 100-cell board with traditional snake/ladder positions
        """
        # Snakes (head -> tail)
        snakes = [
            (99, 54), (70, 55), (52, 42), (25, 2), (95, 72),
            (97, 75), (93, 73), (87, 24), (64, 36), (62, 18)
        ]
        
        # Ladders (bottom -> top)
        ladders = [
            (3, 51), (6, 27), (20, 70), (36, 55), (63, 95),
            (68, 98), (24, 73), (43, 77), (19, 80), (2, 23)
        ]
        
        # Add snakes
        for head, tail in snakes:
            success, msg = self.board.add_snake(head, tail)
            if not success:
                return False, f"Failed to add snake: {msg}"
        
        # Add ladders
        for bottom, top in ladders:
            success, msg = self.board.add_ladder(bottom, top)
            if not success:
                return False, f"Failed to add ladder: {msg}"
        
        return True, "Classic board configured"
    
    def start_game(self) -> Tuple[bool, str]:
        """
        Start the game
        
        Interview Focus: State validation before starting
        """
        with self.lock:
            if self.game_started:
                return False, "Game already started"
            
            if len(self.players) < 2:
                return False, "Need at least 2 players to start"
            
            # Set all players to playing status
            for player in self.players.values():
                player.status = PlayerStatus.PLAYING
            
            self.game_started = True
            self.current_player_index = 0
            
            return True, "Game started!"
    
    # ========================================================================
    # GAMEPLAY
    # ========================================================================
    
    def get_current_player(self) -> Optional[Player]:
        """Get current player whose turn it is"""
        if not self.game_started or self.game_ended:
            return None
        
        player_id = self.player_order[self.current_player_index]
        return self.players[player_id]
    
    def roll_dice_and_move(self) -> Tuple[bool, str]:
        """
        Current player rolls dice and moves
        
        Interview Focus: Core game loop logic
        
        Key Challenges:
        - Handle snake and ladder transitions
        - Check win condition
        - Update statistics
        - Record move history
        """
        with self.lock:
            if not self.game_started:
                return False, "Game not started"
            
            if self.game_ended:
                return False, "Game already ended"
            
            # Get current player
            current_player = self.get_current_player()
            if not current_player:
                return False, "No current player"
            
            # Roll dice
            dice_roll = self.dice.roll()
            start_position = current_player.position
            
            # Calculate new position
            new_position, event = self.board.get_next_position(start_position, dice_roll)
            
            # Update player position
            current_player.move_to(new_position)
            
            # Update statistics based on event
            if event == "snake":
                current_player.hit_snake()
            elif event == "ladder":
                current_player.climb_ladder()
            
            # Check win condition
            if self.board.is_winning_position(new_position):
                current_player.status = PlayerStatus.WON
                self.winner = current_player
                self.game_ended = True
                event = "won"
            
            # Record move
            self.move_counter += 1
            move = GameMove(
                move_number=self.move_counter,
                player_id=current_player.player_id,
                player_name=current_player.name,
                dice_roll=dice_roll,
                start_position=start_position,
                end_position=new_position,
                event=event
            )
            self.moves.append(move)
            
            # Prepare result message
            msg = f"{current_player.name} rolled {dice_roll}: {start_position}→{new_position}"
            
            if event == "snake":
                msg += f" 🐍 (hit snake!)"
            elif event == "ladder":
                msg += f" 🪜 (climbed ladder!)"
            elif event == "won":
                msg += f" 🎉 WON THE GAME!"
            elif event == "boundary":
                msg += f" (exceeded board, stayed at {start_position})"
            
            # Move to next player
            if not self.game_ended:
                self.current_player_index = (self.current_player_index + 1) % len(self.player_order)
            
            return True, msg
    
    def play_full_game(self, max_moves: int = 1000) -> Tuple[bool, str]:
        """
        Play game automatically until someone wins
        
        Interview Focus: Automated gameplay for testing
        """
        with self.lock:
            if not self.game_started:
                return False, "Game not started"
            
            moves_played = 0
            while not self.game_ended and moves_played < max_moves:
                success, msg = self.roll_dice_and_move()
                if not success:
                    return False, msg
                moves_played += 1
            
            if self.game_ended and self.winner:
                return True, f"{self.winner.name} won in {self.move_counter} moves!"
            else:
                return False, f"Game did not finish in {max_moves} moves"
    
    # ========================================================================
    # STATISTICS AND REPORTING
    # ========================================================================
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state
        
        Interview Focus: State snapshot for monitoring
        """
        with self.lock:
            return {
                "game_id": self.game_id,
                "board_size": self.board.size,
                "total_snakes": len(self.board.snakes),
                "total_ladders": len(self.board.ladders),
                "total_players": len(self.players),
                "game_started": self.game_started,
                "game_ended": self.game_ended,
                "total_moves": self.move_counter,
                "current_player": self.get_current_player().name if self.get_current_player() else None,
                "winner": self.winner.name if self.winner else None
            }
    
    def get_player_standings(self) -> List[Dict[str, Any]]:
        """
        Get current player standings
        
        Interview Focus: Sorted leaderboard
        """
        with self.lock:
            standings = []
            
            for player in self.players.values():
                standings.append({
                    "name": player.name,
                    "position": player.position,
                    "moves": player.moves_count,
                    "snakes_hit": player.snakes_hit,
                    "ladders_climbed": player.ladders_climbed,
                    "status": player.status.value
                })
            
            # Sort by position (descending), then by moves (ascending)
            standings.sort(key=lambda x: (-x["position"], x["moves"]))
            
            return standings
    
    def get_move_history(self, last_n: Optional[int] = None) -> List[str]:
        """Get move history (last N moves if specified)"""
        with self.lock:
            moves = self.moves[-last_n:] if last_n else self.moves
            return [str(move) for move in moves]
    
    def get_board_visualization(self) -> str:
        """
        Get simple board visualization
        
        Interview Focus: String representation for debugging
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"BOARD (Size: {self.board.size})")
        lines.append(f"{'='*60}")
        
        # Show snakes
        lines.append(f"\n🐍 Snakes ({len(self.board.snakes)}):")
        for snake in sorted(self.board.snakes.values(), key=lambda s: s.head, reverse=True):
            lines.append(f"  {snake}")
        
        # Show ladders
        lines.append(f"\n🪜 Ladders ({len(self.board.ladders)}):")
        for ladder in sorted(self.board.ladders.values(), key=lambda l: l.bottom):
            lines.append(f"  {ladder}")
        
        # Show player positions
        lines.append(f"\n👥 Players ({len(self.players)}):")
        for player in self.players.values():
            status_icon = "👑" if player.status == PlayerStatus.WON else "▶️" if self.get_current_player() == player else "⏸️"
            lines.append(f"  {status_icon} {player.name}: Position {player.position}")
        
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)


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


def demo_basic_game():
    """Demonstrate basic game flow"""
    print_separator("Basic Snake and Ladder Game")
    
    # Create game
    game = SnakeAndLadderGame(board_size=100)
    
    # Add players
    print("\n🔹 Adding players:")
    game.add_player("Alice")
    game.add_player("Bob")
    game.add_player("Charlie")
    
    # Setup board
    print("\n🔹 Setting up classic board:")
    game.setup_classic_board()
    
    # Show board
    print(game.get_board_visualization())
    
    # Start game
    print("\n🔹 Starting game:")
    success, msg = game.start_game()
    print(f"  {msg}")
    
    # Play a few moves manually
    print("\n🔹 First 10 moves:")
    for i in range(10):
        success, msg = game.roll_dice_and_move()
        if success:
            print(f"  {msg}")
        
        if game.game_ended:
            break
    
    # Show standings
    print("\n🔹 Current standings:")
    for standing in game.get_player_standings():
        print(f"  {standing['name']}: Position {standing['position']} "
              f"({standing['moves']} moves, {standing['snakes_hit']} snakes, "
              f"{standing['ladders_climbed']} ladders)")
    
    return game


def demo_complete_game():
    """Demonstrate complete game play"""
    print_separator("Complete Game Simulation")
    
    # Create game with faster dice (higher values more likely)
    weighted_dice = WeightedDice({1: 0.05, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.3, 6: 0.3})
    game = SnakeAndLadderGame(board_size=100, dice_strategy=weighted_dice)
    
    # Add players
    print("\n🔹 Adding players:")
    game.add_player("Player 1")
    game.add_player("Player 2")
    
    # Setup board
    game.setup_classic_board()
    
    # Start and play full game
    print("\n🔹 Starting game with weighted dice (favors 5-6)...")
    game.start_game()
    
    success, msg = game.play_full_game(max_moves=1000)
    print(f"  {msg}")
    
    # Show final standings
    print("\n🔹 Final standings:")
    for standing in game.get_player_standings():
        print(f"  {standing['name']}: Position {standing['position']} "
              f"({standing['moves']} moves, {standing['snakes_hit']} snakes, "
              f"{standing['ladders_climbed']} ladders) - {standing['status'].upper()}")
    
    # Show last 5 moves
    print("\n🔹 Last 5 moves:")
    for move in game.get_move_history(last_n=5):
        print(f"  {move}")
    
    # Show game statistics
    print("\n🔹 Game statistics:")
    state = game.get_game_state()
    print(f"  Total moves: {state['total_moves']}")
    print(f"  Winner: {state['winner']}")
    print(f"  Board: {state['board_size']} cells, {state['total_snakes']} snakes, {state['total_ladders']} ladders")


def demo_controlled_dice():
    """Demonstrate game with controlled dice for testing"""
    print_separator("Controlled Dice Testing")
    
    # Create game with predetermined dice rolls
    print("\n🔹 Creating game with controlled dice rolls: [6, 6, 5, 1, 6, 6]")
    controlled_dice = ControlledDice([6, 6, 5, 1, 6, 6])
    game = SnakeAndLadderGame(board_size=50, dice_strategy=controlled_dice)
    
    # Add simple snakes and ladders
    game.board.add_snake(25, 5)
    game.board.add_snake(48, 10)
    game.board.add_ladder(8, 30)
    game.board.add_ladder(15, 45)
    
    # Add player
    game.add_player("Test Player")
    game.start_game()
    
    # Play moves
    print("\n🔹 Playing with predetermined rolls:")
    for i in range(6):
        success, msg = game.roll_dice_and_move()
        if success:
            print(f"  {msg}")
        if game.game_ended:
            break
    
    # Show move history
    print("\n🔹 Move history:")
    for move in game.get_move_history():
        print(f"  {move}")


def demo_custom_board():
    """Demonstrate custom board configuration"""
    print_separator("Custom Board Configuration")
    
    # Create small game
    game = SnakeAndLadderGame(board_size=30)
    
    # Add players
    game.add_player("Alice")
    game.add_player("Bob")
    
    # Add custom snakes and ladders
    print("\n🔹 Adding custom snakes and ladders:")
    
    # Snakes
    snakes_config = [(28, 5), (20, 8), (15, 3)]
    for head, tail in snakes_config:
        success, msg = game.board.add_snake(head, tail)
        print(f"  {msg}")
    
    # Ladders
    ladders_config = [(2, 25), (7, 18), (12, 29)]
    for bottom, top in ladders_config:
        success, msg = game.board.add_ladder(bottom, top)
        print(f"  {msg}")
    
    # Show board
    print(game.get_board_visualization())
    
    # Play game
    print("\n🔹 Playing game:")
    game.start_game()
    game.play_full_game(max_moves=500)
    
    # Show results
    print("\n🔹 Final results:")
    for standing in game.get_player_standings():
        print(f"  {standing['name']}: {standing['status'].upper()} at position {standing['position']}")


def demo_statistics():
    """Demonstrate game statistics and analytics"""
    print_separator("Game Statistics and Analytics")
    
    print("\n🔹 Running 5 simulated games to gather statistics...")
    
    results = []
    
    for game_num in range(5):
        game = SnakeAndLadderGame(board_size=100)
        game.add_player(f"Player A")
        game.add_player(f"Player B")
        game.setup_classic_board()
        game.start_game()
        game.play_full_game(max_moves=1000)
        
        state = game.get_game_state()
        winner_stats = None
        for standing in game.get_player_standings():
            if standing['status'] == 'won':
                winner_stats = standing
                break
        
        results.append({
            "game": game_num + 1,
            "total_moves": state['total_moves'],
            "winner": winner_stats['name'] if winner_stats else "None",
            "winner_moves": winner_stats['moves'] if winner_stats else 0,
            "snakes_hit": winner_stats['snakes_hit'] if winner_stats else 0,
            "ladders_climbed": winner_stats['ladders_climbed'] if winner_stats else 0
        })
    
    # Display statistics
    print("\n🔹 Game Statistics:")
    print(f"  {'Game':<6} {'Moves':<8} {'Winner':<12} {'Snakes':<8} {'Ladders'}")
    print(f"  {'-'*50}")
    
    for result in results:
        print(f"  {result['game']:<6} {result['total_moves']:<8} {result['winner']:<12} "
              f"{result['snakes_hit']:<8} {result['ladders_climbed']}")
    
    # Calculate averages
    avg_moves = sum(r['total_moves'] for r in results) / len(results)
    avg_snakes = sum(r['snakes_hit'] for r in results) / len(results)
    avg_ladders = sum(r['ladders_climbed'] for r in results) / len(results)
    
    print(f"\n🔹 Averages across {len(results)} games:")
    print(f"  Average total moves: {avg_moves:.1f}")
    print(f"  Average snakes hit: {avg_snakes:.1f}")
    print(f"  Average ladders climbed: {avg_ladders:.1f}")


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  SNAKE AND LADDER GAME - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Classic board game, Strategy pattern, Multi-player")
    print("="*70)
    
    demo_basic_game()
    demo_complete_game()
    demo_controlled_dice()
    demo_custom_board()
    demo_statistics()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python 12_snake_ladder_game.py
    """
    run_demo()
