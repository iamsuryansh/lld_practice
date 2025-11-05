# Chess Game - Interview Preparation Guide

**Target Audience**: Software Engineers with 2-5 years of experience  
**Focus**: Object-oriented design, piece movement validation, game state management  
**Estimated Study Time**: 3-4 hours

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

Design a chess game system that can:
- Represent an 8x8 chess board with all 6 piece types
- Validate piece-specific moves (King, Queen, Rook, Bishop, Knight, Pawn)
- Handle special moves: Castling, En Passant, Pawn Promotion
- Detect game states: Check, Checkmate, Stalemate
- Support turn-based gameplay with move history

**Core Challenge**: How do you design a flexible piece movement system while enforcing complex chess rules and validating game states?

---

## Step-by-Step Implementation Guide

### Phase 1: Board Representation (10-15 minutes)

**What to do**:
```python
class Position:
    def __init__(self, row: int, col: int):
        self.row = row  # 0-7 (0=rank 8, 7=rank 1)
        self.col = col  # 0-7 (0=file a, 7=file h)
    
    def to_algebraic(self) -> str:
        return f"{chr(97 + self.col)}{8 - self.row}"

class Board:
    def __init__(self):
        self.grid = [[None] * 8 for _ in range(8)]
```

**Why this approach**:
- **2D array**: O(1) access by position, simple to visualize
- **0-indexed coordinates**: Standard programming convention
- **Algebraic notation**: Standard chess notation (e2, e4, etc.)
- **Alternative**: Bitboards (64-bit integers) for competitive chess engines, but adds complexity

**Common mistake**: Using 1-indexed coordinates or starting from bottom-left instead of top-left.

---

### Phase 2: Piece Hierarchy with Strategy Pattern (15-20 minutes)

**What to do**:
```python
class Piece(ABC):
    def __init__(self, color: Color, position: Position):
        self.color = color
        self.position = position
        self.has_moved = False  # For castling and pawn moves
    
    @abstractmethod
    def get_possible_moves(self, board: Board) -> List[Position]:
        pass

class Rook(Piece):
    def get_possible_moves(self, board: Board) -> List[Position]:
        # Move horizontally and vertically
        return self._get_linear_moves(board, [(-1,0), (1,0), (0,-1), (0,1)])
```

**Why Strategy Pattern**:
- **Polymorphism**: Each piece encapsulates its own movement logic
- **Open/Closed Principle**: Easy to add new piece types without modifying existing code
- **Testability**: Can test each piece type independently

**Interview Insight**: Explain that Strategy Pattern is perfect here because pieces have radically different movement behaviors that can't be parameterized easily.

---

### Phase 3: Move Validation Algorithm (20-25 minutes)

**What to do**:
```python
def _is_legal_move(self, move: Move) -> bool:
    """Check if move doesn't leave king in check"""
    # 1. Clone board to simulate move
    test_board = self.board.clone()
    
    # 2. Execute move on cloned board
    test_board.move_piece(move.start, move.end)
    
    # 3. Find king position after move
    king_pos = test_board.find_king(move.piece.color)
    
    # 4. Check if king is under attack
    return not test_board.is_position_under_attack(king_pos, move.piece.color)
```

**Why Clone-and-Validate**:
- Can't modify actual game state during validation
- Need to see future board state to detect check
- Immutability prevents bugs from partial state changes

**Critical Detail**: This is O(n²) because checking if king is under attack requires scanning all opponent pieces. Optimization possible with piece position tracking.

**When it fails**: Very deep validation (e.g., 3-fold repetition) requires full game history analysis.

---

### Phase 4: Special Moves Implementation (20-30 minutes)

**What to do**:
```python
# Castling
def _get_castling_moves(self, board: Board) -> List[Position]:
    if self.has_moved:
        return []
    
    # Check kingside: king moves 2 squares right
    if self._can_castle_kingside(board):
        moves.append(Position(self.position.row, 6))
    
    # Check queenside: king moves 2 squares left
    if self._can_castle_queenside(board):
        moves.append(Position(self.position.row, 2))

# En Passant
def _get_en_passant_moves(self, board: Board) -> List[Position]:
    if not board.last_move:
        return []
    
    last_move = board.last_move
    # Check if last move was pawn double-move adjacent to this pawn
    if (last_move.piece.piece_type == PieceType.PAWN and
        abs(last_move.end.row - last_move.start.row) == 2 and
        abs(last_move.end.col - self.position.col) == 1):
        # Can capture en passant
        return [Position(last_move.end.row + direction, last_move.end.col)]
```

**State Transition Flow**:
```
INITIAL → King/Rook unmoved → Castling available
        → King/Rook moves → Castling lost forever

LAST_MOVE → Pawn double-move → En passant available next turn only
          → Any other move → En passant opportunity lost
```

**Why this pattern**:
- Special moves have timing dependencies (must track `has_moved`, `last_move`)
- Castling requires checking 5 conditions simultaneously
- En passant is only legal immediately after opponent's double-move

**Interview Tip**: Mention that castling is the only move in chess that moves two pieces, making it unique algorithmically.

---

### Phase 5: Game State Detection (15-20 minutes)

**What to do**:
```python
def _update_game_state(self) -> None:
    king_pos = self.board.find_king(self.current_turn)
    in_check = self.board.is_position_under_attack(king_pos, self.current_turn)
    has_legal_moves = self._has_legal_moves()
    
    if in_check:
        if has_legal_moves:
            self.game_state = GameState.CHECK
        else:
            self.game_state = GameState.CHECKMATE
    else:
        if has_legal_moves:
            self.game_state = GameState.ACTIVE
        else:
            self.game_state = GameState.STALEMATE
```

**Error Recovery Strategy**:
- Validate every move before execution
- Use immutable move validation (clone board)
- Track move history for undo functionality
- Return `(bool, str)` tuples for clear error messages

---

## Critical Knowledge Points

### 1. Why Strategy Pattern for Pieces?

**Without Pattern**:
```python
def get_moves(piece_type, position, board):
    if piece_type == "ROOK":
        # Rook logic
    elif piece_type == "BISHOP":
        # Bishop logic
    elif piece_type == "KNIGHT":
        # Knight logic
    # ... 20 more conditions
```

**With Pattern**:
```python
class Rook(Piece):
    def get_possible_moves(self, board):
        return self._get_linear_moves(board, [(-1,0), (1,0), (0,-1), (0,1)])

# Just call: piece.get_possible_moves(board)
```

**Benefits**:
- **Eliminates giant switch statements**: Each piece knows its own behavior
- **Type safety**: Compile-time checking of piece types
- **Extensibility**: Add new piece types (fairy chess pieces) without touching existing code
- **Testability**: Mock individual piece types easily

---

### 2. Board Representation Tradeoffs

**2D Array (Our Choice)**:
```python
board = [[None] * 8 for _ in range(8)]
piece = board[row][col]  # O(1) access
```

**Time**: O(1) access  
**Space**: O(64) = O(1) for chess

**Why it works**: Chess board is fixed 8×8, simple to understand and debug

**Alternative: Bitboards**:
```python
white_pawns = 0x000000000000FF00  # 64-bit integer
white_rooks = 0x0000000000000081
```

**Time**: O(1) access, O(1) move generation with bit operations  
**Space**: O(1), extremely compact

**When to use**: Competitive chess engines where performance is critical. Bitboards enable parallel processing of multiple pieces using SIMD instructions.

---

### 3. Check Detection Algorithm

**Algorithm**:
```python
def is_position_under_attack(position: Position, by_color: Color) -> bool:
    # Iterate all opponent pieces
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color == by_color.opposite():
                # Special case: Pawns attack diagonally but move forward
                if piece.piece_type == PAWN:
                    if position in piece.get_attack_squares():
                        return True
                else:
                    if position in piece.get_possible_moves(board):
                        return True
    return False
```

**Time**: O(n²) where n=8, so O(64) = O(1) for chess  
**Space**: O(1)

**Why it works**: Must check all opponent pieces to see if any can attack the target square.

**Optimization**:
```python
# Track piece positions for O(1) lookup
self.piece_positions = {
    Color.WHITE: {PieceType.ROOK: [pos1, pos2], ...},
    Color.BLACK: {PieceType.QUEEN: [pos1], ...}
}
```

**Time**: O(p) where p = number of opponent pieces  
**Space**: O(32) = O(1) for maximum pieces

---

### 4. Castling Validation

**Problem**: Castling has 5 simultaneous requirements:
1. King hasn't moved
2. Rook hasn't moved
3. No pieces between king and rook
4. King not currently in check
5. King doesn't move through check

**Solution**:
```python
def _can_castle_kingside(self, board: Board) -> bool:
    if self.has_moved:
        return False
    
    # Check rook hasn't moved
    rook = board.get_piece(Position(self.position.row, 7))
    if not rook or rook.has_moved:
        return False
    
    # Check squares between are empty
    for col in range(5, 7):
        if board.get_piece(Position(self.position.row, col)):
            return False
    
    # Check king doesn't move through check
    for col in range(4, 7):
        if board.is_position_under_attack(Position(self.position.row, col), self.color):
            return False
    
    return True
```

**Key Insight**: Track `has_moved` flag on King and Rooks from game start. Once set to True, it can never be undone.

---

## Expected Interview Questions & Answers

### Q1: How would you scale this to support online multiplayer chess?

**Answer**:
To support online multiplayer, I'd separate the game logic from network concerns using a client-server architecture:

**Server-Side**:
- Maintain authoritative game state
- Validate all moves server-side (never trust client)
- Use WebSockets for real-time move synchronization
- Store game state in database (Redis for active games, PostgreSQL for history)

**Implementation**:
```python
class ChessServer:
    def __init__(self):
        self.active_games = {}  # game_id -> ChessGame
        self.redis_client = Redis()
    
    async def handle_move(self, game_id: str, player_id: str, move: dict):
        # 1. Validate player's turn
        game = self.active_games[game_id]
        if game.current_turn != player_id:
            return {"error": "Not your turn"}
        
        # 2. Validate and execute move
        success, message = game.make_move(move['from'], move['to'])
        
        # 3. Broadcast to both players
        await self.broadcast_game_state(game_id, game.get_state())
        
        # 4. Persist to Redis
        self.redis_client.set(f"game:{game_id}", game.serialize())
        
        return {"success": success, "message": message}
```

**Key Points**:
1. **Authoritative server**: Client is just a view, server holds truth
2. **WebSockets**: Low-latency bidirectional communication
3. **State persistence**: Redis for fast access, PostgreSQL for durability
4. **Latency optimization**: Optimistic client-side updates with rollback on server rejection

**Follow-up**: How do you handle disconnections? Use heartbeat pings, reconnection tokens, and pause game on disconnect with timeout.

---

### Q2: How would you implement an AI opponent?

**Answer**:
I'd use the Minimax algorithm with Alpha-Beta pruning for move selection:

**Basic Minimax**:
```python
def minimax(board: Board, depth: int, is_maximizing: bool) -> Tuple[int, Move]:
    if depth == 0 or game_over:
        return evaluate_board(board), None
    
    if is_maximizing:
        max_eval = float('-inf')
        best_move = None
        for move in get_all_legal_moves(board):
            test_board = board.clone()
            test_board.execute_move(move)
            eval_score, _ = minimax(test_board, depth - 1, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        # Minimizing player logic (opponent)
        min_eval = float('inf')
        # ... similar logic
        return min_eval, best_move

def evaluate_board(board: Board) -> int:
    """Heuristic board evaluation"""
    score = 0
    piece_values = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 100}
    
    for piece in board.get_all_pieces():
        value = piece_values[piece.type]
        score += value if piece.color == AI_COLOR else -value
    
    return score
```

**Optimizations**:
1. **Alpha-Beta Pruning**: Skip branches that won't affect final decision (reduce from O(b^d) to O(b^(d/2)))
2. **Move Ordering**: Check captures and checks first to maximize pruning
3. **Transposition Tables**: Cache evaluated positions to avoid recalculation
4. **Iterative Deepening**: Start with depth 1, gradually increase for better move ordering

**Follow-up**: For stronger AI, use opening books, endgame tablebases, and machine learning position evaluation (like Stockfish or AlphaZero).

---

### Q3: How do you detect stalemate vs checkmate?

**Answer**:
The key difference is whether the king is currently in check:

**Detection Logic**:
```python
def detect_game_end(self) -> GameState:
    king_pos = self.board.find_king(self.current_turn)
    in_check = self.board.is_position_under_attack(king_pos, self.current_turn)
    has_legal_moves = self._has_legal_moves()
    
    # Critical distinction:
    if not has_legal_moves:
        if in_check:
            return CHECKMATE  # King in check + no moves = checkmate
        else:
            return STALEMATE  # King safe + no moves = stalemate
    
    return CHECK if in_check else ACTIVE
```

**Why this works**:
- **Checkmate**: Player is in immediate danger (check) and cannot escape
- **Stalemate**: Player is safe but any move would put them in danger (draw)

**Edge Case Example**:
```
Position: King at h8, opponent Queen at g7, opponent King at f6
- Black king not in check (can't be captured this turn)
- All moves (g8, g7, h7) would put king in check
- Result: Stalemate (draw)
```

---

### Q4: How would you implement move undo/redo functionality?

**Answer**:
Maintain a move history stack with enough information to reverse each move:

**Implementation**:
```python
@dataclass
class MoveRecord:
    move: Move
    captured_piece: Optional[Piece]
    previous_game_state: GameState
    castling_rights_before: Dict[Color, Dict[str, bool]]
    en_passant_target_before: Optional[Position]

class ChessGame:
    def __init__(self):
        self.move_history: List[MoveRecord] = []
        self.redo_stack: List[MoveRecord] = []
    
    def undo_move(self) -> bool:
        if not self.move_history:
            return False
        
        record = self.move_history.pop()
        
        # Reverse the move
        self.board.move_piece(record.move.end, record.move.start)
        
        # Restore captured piece
        if record.captured_piece:
            self.board.set_piece(record.move.end, record.captured_piece)
        
        # Restore game state
        self.game_state = record.previous_game_state
        self.current_turn = self.current_turn.opposite()
        
        # Add to redo stack
        self.redo_stack.append(record)
        
        return True
    
    def redo_move(self) -> bool:
        if not self.redo_stack:
            return False
        
        record = self.redo_stack.pop()
        self._execute_move(record.move)
        self.move_history.append(record)
        return True
```

**Key Points**:
1. Store complete game state before each move
2. Redo stack cleared on new moves (branching)
3. Special moves (castling, en passant) need special reversal logic

---

### Q5: How would you optimize move generation for performance?

**Answer**:
Several optimization strategies for high-performance chess engines:

**1. Piece Position Tracking**:
```python
class Board:
    def __init__(self):
        self.grid = [[None] * 8 for _ in range(8)]
        # Track piece positions to avoid scanning entire board
        self.white_pieces: Set[Position] = set()
        self.black_pieces: Set[Position] = set()
        self.king_positions = {Color.WHITE: None, Color.BLACK: None}
    
    def get_all_legal_moves(self, color: Color) -> List[Move]:
        moves = []
        # Only iterate actual pieces, not empty squares
        for pos in self.white_pieces if color == WHITE else self.black_pieces:
            piece = self.grid[pos.row][pos.col]
            moves.extend(piece.get_possible_moves(self))
        return moves
```

**2. Move Generation Order** (for Alpha-Beta pruning):
```python
def order_moves(moves: List[Move]) -> List[Move]:
    """Order moves for better pruning"""
    def move_priority(move: Move) -> int:
        score = 0
        if move.captured_piece:
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            score += 10 * piece_value(move.captured_piece)
            score -= piece_value(move.piece)
        if move.is_check:
            score += 50
        if move.is_castling:
            score += 30
        return score
    
    return sorted(moves, key=move_priority, reverse=True)
```

**3. Lazy Move Validation**:
```python
def get_pseudo_legal_moves(piece: Piece) -> List[Position]:
    """Generate moves without checking if they leave king in check"""
    # Much faster, validate only when needed
    return piece._generate_moves_no_validation()

def filter_legal_moves(moves: List[Move]) -> List[Move]:
    """Only validate moves that will actually be played"""
    return [m for m in moves if not self._leaves_king_in_check(m)]
```

**Alternative approach**: Use bitboards for professional engines - enables parallel move generation with SIMD instructions.

---

### Q6: How would you handle three-fold repetition and fifty-move rule?

**Answer**:
These draw conditions require tracking full game history:

**Implementation**:
```python
class ChessGame:
    def __init__(self):
        self.position_history: Dict[str, int] = {}  # position_hash -> count
        self.halfmove_clock = 0  # Moves since capture or pawn move
    
    def make_move(self, start: str, end: str):
        # ... execute move
        
        # Update fifty-move rule counter
        if move.captured_piece or move.piece.piece_type == PAWN:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        # Check fifty-move rule
        if self.halfmove_clock >= 100:  # 50 full moves = 100 half moves
            self.game_state = GameState.DRAW
            return True, "Draw by fifty-move rule"
        
        # Update position history for three-fold repetition
        position_hash = self._hash_position()
        self.position_history[position_hash] = \
            self.position_history.get(position_hash, 0) + 1
        
        # Check three-fold repetition
        if self.position_history[position_hash] >= 3:
            self.game_state = GameState.DRAW
            return True, "Draw by three-fold repetition"
    
    def _hash_position(self) -> str:
        """Hash current position including castling rights and en passant"""
        pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.grid[row][col]
                if piece:
                    pieces.append(f"{piece.color.value}{piece.piece_type.value}{row}{col}")
        
        # Include castling rights and en passant in hash
        castling = f"{self.can_castle_kingside}{self.can_castle_queenside}"
        en_passant = str(self.board.en_passant_target) if self.board.en_passant_target else ""
        
        return "|".join(pieces) + f"|{castling}|{en_passant}"
```

---

### Q7: How would you implement a chess puzzle validator?

**Answer**:
Validate that a sequence of moves leads to the expected outcome (checkmate, win material, etc.):

**Implementation**:
```python
class ChessPuzzle:
    def __init__(self, fen: str, solution_moves: List[str], theme: str):
        self.initial_position = fen
        self.solution = solution_moves
        self.theme = theme  # "mate_in_2", "fork", "pin", etc.
    
    def validate_solution(self, user_moves: List[str]) -> Tuple[bool, str]:
        game = ChessGame()
        game.load_from_fen(self.initial_position)
        
        # Execute user moves
        for i, move in enumerate(user_moves):
            success, msg = game.make_move_algebraic(move)
            if not success:
                return False, f"Invalid move at position {i+1}: {msg}"
            
            # Check if expected move
            if i < len(self.solution):
                if move != self.solution[i]:
                    return False, f"Expected {self.solution[i]}, got {move}"
        
        # Validate outcome matches puzzle theme
        if self.theme.startswith("mate_in_"):
            moves = int(self.theme.split("_")[2])
            if game.game_state == GameState.CHECKMATE and len(user_moves) <= moves * 2:
                return True, "Correct! Checkmate achieved"
            return False, "Checkmate not achieved in required moves"
        
        return True, "Puzzle solved!"
```

---

## Testing Strategy

### Unit Tests

**Test piece movement independently**:
```python
def test_rook_movement():
    board = Board()
    rook = Rook(Color.WHITE, Position(0, 0))
    board.set_piece(Position(0, 0), rook)
    
    moves = rook.get_possible_moves(board)
    
    # Should move along row 0 and column 0
    assert Position(0, 7) in moves  # Right
    assert Position(7, 0) in moves  # Down
    assert Position(0, 1) not in moves  # Can't jump to occupied square

def test_knight_jumping():
    board = Board()
    board.setup_initial_position()
    knight = board.get_piece(Position(7, 1))  # White knight
    
    moves = knight.get_possible_moves(board)
    
    # Knight should jump over pawns
    assert Position(5, 0) in moves
    assert Position(5, 2) in moves
    assert len(moves) == 2  # Only 2 moves from starting position
```

**Test special moves**:
```python
def test_castling_kingside():
    game = ChessGame()
    game.board = Board()
    
    # Setup castling position
    game.board.set_piece(Position(7, 4), King(Color.WHITE, Position(7, 4)))
    game.board.set_piece(Position(7, 7), Rook(Color.WHITE, Position(7, 7)))
    
    success, _ = game.make_move("e1", "g1")
    
    assert success
    assert game.board.get_piece(Position(7, 6)).piece_type == PieceType.KING
    assert game.board.get_piece(Position(7, 5)).piece_type == PieceType.ROOK

def test_en_passant():
    game = ChessGame()
    # Setup: White pawn at e5, black pawn moves d7->d5
    # ... setup code ...
    
    game.make_move("d7", "d5")  # Black pawn double-move
    success, _ = game.make_move("e5", "d6")  # En passant capture
    
    assert success
    assert game.board.get_piece(Position(2, 3)) is None  # Captured pawn removed
```

---

### Integration Tests

**Test complete game flow**:
```python
def test_scholars_mate():
    game = ChessGame()
    
    moves = [
        ("e2", "e4"), ("e7", "e5"),
        ("f1", "c4"), ("b8", "c6"),
        ("d1", "h5"), ("g8", "f6"),
        ("h5", "f7")
    ]
    
    for start, end in moves[:-1]:
        success, _ = game.make_move(start, end)
        assert success
        assert game.game_state != GameState.CHECKMATE
    
    # Final move should be checkmate
    success, msg = game.make_move(moves[-1][0], moves[-1][1])
    assert success
    assert game.game_state == GameState.CHECKMATE
    assert "Checkmate" in msg
```

**Test stalemate detection**:
```python
def test_stalemate():
    game = ChessGame()
    game.board = Board()
    
    # Setup stalemate position
    game.board.set_piece(Position(0, 7), King(Color.BLACK, Position(0, 7)))
    game.board.set_piece(Position(2, 6), Queen(Color.WHITE, Position(2, 6)))
    game.board.set_piece(Position(7, 0), King(Color.WHITE, Position(7, 0)))
    game.current_turn = Color.WHITE
    
    game.make_move("g6", "g7")
    
    assert game.game_state == GameState.STALEMATE
```

---

### Load Testing

**Concurrent games simulation**:
```python
import threading
import time

def test_concurrent_games():
    games = [ChessGame(f"game_{i}") for i in range(100)]
    
    def play_random_game(game):
        while game.game_state == GameState.ACTIVE:
            # Get random legal move
            all_moves = game._get_all_legal_moves()
            if not all_moves:
                break
            move = random.choice(all_moves)
            game.make_move(move.start.to_algebraic(), move.end.to_algebraic())
    
    threads = [threading.Thread(target=play_random_game, args=(g,)) for g in games]
    
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    print(f"100 games completed in {elapsed:.2f} seconds")
```

---

## Production Considerations

### 1. Persistence

**Current implementation**: In-memory only  
**Production needs**: Database persistence with game replay

```python
class ChessGameRepository:
    def __init__(self, db: Database):
        self.db = db
    
    def save_game(self, game: ChessGame):
        """Save game state to database"""
        self.db.execute("""
            INSERT INTO games (game_id, current_state, move_history, created_at)
            VALUES (?, ?, ?, ?)
        """, (game.game_id, game.serialize(), json.dumps(game.move_history), datetime.now()))
    
    def load_game(self, game_id: str) -> ChessGame:
        """Load game from database"""
        row = self.db.query_one("SELECT * FROM games WHERE game_id = ?", (game_id,))
        game = ChessGame.deserialize(row['current_state'])
        game.move_history = json.loads(row['move_history'])
        return game
    
    def get_move_history(self, game_id: str) -> List[str]:
        """Get game moves in PGN format"""
        game = self.load_game(game_id)
        return [move.to_algebraic() for move in game.board.move_history]
```

---

### 2. Monitoring & Alerts

**Implement health checks**:
```python
class ChessGameMonitor:
    def __init__(self):
        self.metrics = {
            'games_active': 0,
            'moves_per_second': 0,
            'invalid_moves_count': 0,
            'average_game_length': 0
        }
    
    def record_move(self, game_id: str, success: bool, duration_ms: float):
        """Record move metrics"""
        if not success:
            self.metrics['invalid_moves_count'] += 1
        
        self.metrics['moves_per_second'] = 1000 / duration_ms
        
        # Alert on anomalies
        if duration_ms > 1000:
            self.alert(f"Slow move validation: {duration_ms}ms in game {game_id}")
    
    def check_game_health(self, game: ChessGame) -> HealthStatus:
        """Validate game state consistency"""
        issues = []
        
        # Check board integrity
        if len(game.board.get_all_pieces()) > 32:
            issues.append("Too many pieces on board")
        
        # Check king presence
        if not game.board.find_king(Color.WHITE):
            issues.append("White king missing")
        
        # Check move history consistency
        if len(game.board.move_history) != game.move_count:
            issues.append("Move history mismatch")
        
        return HealthStatus(healthy=len(issues) == 0, issues=issues)
```

---

### 3. Security

**Key concerns**:
1. **Move validation**: Always validate on server (never trust client)
2. **Rate limiting**: Prevent move spam
3. **Input sanitization**: Validate algebraic notation format

```python
class SecureChessGame(ChessGame):
    def __init__(self, game_id: str, white_player_id: str, black_player_id: str):
        super().__init__(game_id)
        self.white_player_id = white_player_id
        self.black_player_id = black_player_id
        self.move_timestamps = []
    
    def make_move_authenticated(self, player_id: str, start: str, end: str):
        # 1. Verify player identity
        expected_player = (self.white_player_id if self.current_turn == Color.WHITE 
                          else self.black_player_id)
        if player_id != expected_player:
            return False, "Not your turn"
        
        # 2. Rate limiting (max 1 move per second)
        now = time.time()
        if self.move_timestamps and now - self.move_timestamps[-1] < 1.0:
            return False, "Move too fast"
        
        # 3. Input validation
        if not self._is_valid_notation(start) or not self._is_valid_notation(end):
            return False, "Invalid notation format"
        
        # 4. Execute move with full validation
        success, message = self.make_move(start, end)
        
        if success:
            self.move_timestamps.append(now)
        
        return success, message
    
    def _is_valid_notation(self, notation: str) -> bool:
        """Validate algebraic notation format"""
        return (len(notation) == 2 and
                notation[0] in 'abcdefgh' and
                notation[1] in '12345678')
```

---

### 4. Scalability

**Managing distributed games**:
```python
class DistributedChessServer:
    def __init__(self):
        self.redis = Redis()
        self.game_nodes = []  # List of game server nodes
    
    def route_game(self, game_id: str) -> str:
        """Consistent hashing for game routing"""
        # Route games to specific servers based on game_id hash
        node_index = hash(game_id) % len(self.game_nodes)
        return self.game_nodes[node_index]
    
    def sync_game_state(self, game: ChessGame):
        """Sync game state across nodes using Redis"""
        self.redis.set(
            f"game:{game.game_id}",
            game.serialize(),
            ex=3600  # Expire after 1 hour of inactivity
        )
    
    def handle_move_distributed(self, game_id: str, player_id: str, move: dict):
        """Handle move in distributed environment"""
        # 1. Acquire lock on game
        lock_key = f"lock:game:{game_id}"
        with self.redis.lock(lock_key, timeout=5):
            # 2. Load game state from Redis
            game_data = self.redis.get(f"game:{game_id}")
            game = ChessGame.deserialize(game_data)
            
            # 3. Execute move
            success, message = game.make_move_authenticated(player_id, move['from'], move['to'])
            
            # 4. Save updated state
            if success:
                self.sync_game_state(game)
            
            return success, message
```

---

## Summary

### Do's ✅
- Use Strategy Pattern for piece movement (polymorphism)
- Validate moves by simulating on cloned board
- Track `has_moved` flag for castling eligibility
- Store complete move history for undo/analysis
- Detect checkmate vs stalemate correctly (king in check vs not)
- Handle all special moves (castling, en passant, promotion)

### Don'ts ❌
- Don't use giant switch statements for piece movement
- Don't modify game state during move validation
- Don't trust client-side validation (always validate server-side)
- Don't forget pawn attack squares differ from move squares
- Don't skip en passant timing validation (only valid immediately after double-move)

### Key Takeaways
1. **Design Pattern Choice**: Strategy Pattern is perfect for radically different behaviors (piece movements)
2. **Validation Strategy**: Clone-and-validate prevents partial state corruption
3. **Special Moves Complexity**: Castling and en passant have timing dependencies requiring careful state tracking
4. **Performance Optimization**: Track piece positions instead of scanning entire board (O(pieces) vs O(64))
5. **Game State Detection**: Checkmate = check + no legal moves; Stalemate = safe + no legal moves

---

**Time to Master**: 3-4 hours  
**Difficulty**: Medium  
**Key Patterns**: Strategy, Template Method  
**Critical Skills**: OOP design, move validation algorithms, game state management
