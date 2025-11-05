"""
Chess Game - Single File Implementation
For coding interviews and production-ready reference

Features:
- Complete chess board representation (8x8 grid)
- All 6 piece types with valid move logic (King, Queen, Rook, Bishop, Knight, Pawn)
- Special moves: Castling, En Passant, Pawn Promotion
- Game state management: Check, Checkmate, Stalemate detection
- Move validation and turn-based gameplay
- Move history tracking with algebraic notation

Interview Focus:
- Object-oriented design with inheritance
- Strategy pattern for piece movement
- State pattern for game phases
- Complex rule validation and edge cases
- Board representation techniques
- Algorithm optimization for move validation

Author: Interview Prep
Date: 2024
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import time


# ============================================================================
# SECTION 1: MODELS - Core data classes and enums
# ============================================================================

class PieceType(Enum):
    """Chess piece types"""
    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = "P"


class Color(Enum):
    """Player colors"""
    WHITE = "white"
    BLACK = "black"
    
    def opposite(self) -> 'Color':
        """Return opposite color"""
        return Color.BLACK if self == Color.WHITE else Color.WHITE


class GameState(Enum):
    """Game state tracking"""
    ACTIVE = "active"
    CHECK = "check"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW = "draw"


@dataclass
class Position:
    """
    Board position using 0-indexed coordinates
    
    Interview Focus: Why use dataclass? Immutability and hash support
    
    Coordinate system:
    - row: 0 (top/rank 8) to 7 (bottom/rank 1)
    - col: 0 (left/file a) to 7 (right/file h)
    """
    row: int
    col: int
    
    def is_valid(self) -> bool:
        """Check if position is within board bounds"""
        return 0 <= self.row < 8 and 0 <= self.col < 8
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col
    
    def __hash__(self) -> int:
        return hash((self.row, self.col))
    
    def to_algebraic(self) -> str:
        """Convert to chess notation (e.g., 'e4')"""
        files = 'abcdefgh'
        ranks = '87654321'
        return f"{files[self.col]}{ranks[self.row]}"
    
    @staticmethod
    def from_algebraic(notation: str) -> 'Position':
        """Create position from algebraic notation"""
        files = 'abcdefgh'
        ranks = '87654321'
        col = files.index(notation[0])
        row = ranks.index(notation[1])
        return Position(row, col)


@dataclass
class Move:
    """
    Represents a chess move
    
    Interview Focus: How do you represent special moves?
    """
    start: Position
    end: Position
    piece: 'Piece'
    captured_piece: Optional['Piece'] = None
    is_castling: bool = False
    is_en_passant: bool = False
    promotion_type: Optional[PieceType] = None
    
    def to_algebraic(self) -> str:
        """Convert move to algebraic notation"""
        notation = ""
        
        # Add piece symbol (except for pawns)
        if self.piece.piece_type != PieceType.PAWN:
            notation += self.piece.piece_type.value
        
        # Add capture symbol
        if self.captured_piece or self.is_en_passant:
            if self.piece.piece_type == PieceType.PAWN:
                notation += self.start.to_algebraic()[0]  # Starting file for pawn captures
            notation += "x"
        
        # Add destination
        notation += self.end.to_algebraic()
        
        # Add promotion
        if self.promotion_type:
            notation += f"={self.promotion_type.value}"
        
        # Add castling notation
        if self.is_castling:
            if self.end.col > self.start.col:
                return "O-O"  # Kingside
            else:
                return "O-O-O"  # Queenside
        
        return notation


# ============================================================================
# SECTION 2: PIECE HIERARCHY - Strategy pattern for movement
# ============================================================================

class Piece(ABC):
    """
    Abstract base class for chess pieces
    
    Strategy Pattern: Each piece encapsulates its own movement logic
    
    Interview Focus: Why abstract base class? Polymorphic behavior for different pieces
    """
    
    def __init__(self, color: Color, position: Position):
        self.color = color
        self.position = position
        self.has_moved = False  # Track for castling and pawn double-move
        self.piece_type: PieceType = None  # Set in subclasses
    
    @abstractmethod
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Get all possible moves (not considering check)
        
        Interview Focus: How do you implement piece-specific movement?
        Time Complexity: Varies by piece (O(1) to O(n))
        """
        pass
    
    def can_move_to(self, board: 'Board', target: Position) -> bool:
        """Check if piece can move to target position"""
        return target in self.get_possible_moves(board)
    
    def _get_linear_moves(self, board: 'Board', directions: List[Tuple[int, int]]) -> List[Position]:
        """
        Helper for pieces that move in straight lines (Rook, Bishop, Queen)
        
        Key Insight: Reusable logic for sliding pieces
        Time Complexity: O(k*d) where k=directions, d=distance
        """
        moves = []
        for dr, dc in directions:
            row, col = self.position.row, self.position.col
            while True:
                row += dr
                col += dc
                new_pos = Position(row, col)
                
                if not new_pos.is_valid():
                    break
                
                target_piece = board.get_piece(new_pos)
                if target_piece is None:
                    moves.append(new_pos)
                elif target_piece.color != self.color:
                    moves.append(new_pos)  # Can capture
                    break
                else:
                    break  # Blocked by own piece
        
        return moves
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        color_prefix = "W" if self.color == Color.WHITE else "B"
        return f"{color_prefix}{self.piece_type.value}"


class King(Piece):
    """
    King piece - moves one square in any direction
    
    Special moves: Castling
    
    Interview Focus: How do you handle castling requirements?
    """
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.KING
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Get king moves (one square in any direction)
        
        Time Complexity: O(1) - always checks 8 squares
        """
        moves = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            new_pos = Position(self.position.row + dr, self.position.col + dc)
            if not new_pos.is_valid():
                continue
            
            target_piece = board.get_piece(new_pos)
            if target_piece is None or target_piece.color != self.color:
                moves.append(new_pos)
        
        # Add castling moves
        moves.extend(self._get_castling_moves(board))
        
        return moves
    
    def _get_castling_moves(self, board: 'Board') -> List[Position]:
        """
        Get valid castling positions
        
        Castling requirements:
        1. King hasn't moved
        2. Rook hasn't moved
        3. No pieces between king and rook
        4. King not in check
        5. King doesn't move through check
        """
        if self.has_moved:
            return []
        
        moves = []
        row = self.position.row
        
        # Kingside castling (O-O)
        kingside_rook_pos = Position(row, 7)
        kingside_rook = board.get_piece(kingside_rook_pos)
        if (kingside_rook and 
            kingside_rook.piece_type == PieceType.ROOK and 
            not kingside_rook.has_moved):
            
            # Check if squares between are empty
            if all(board.get_piece(Position(row, c)) is None for c in range(5, 7)):
                # Check if king doesn't move through check
                if not any(board.is_position_under_attack(Position(row, c), self.color) 
                          for c in range(4, 7)):
                    moves.append(Position(row, 6))
        
        # Queenside castling (O-O-O)
        queenside_rook_pos = Position(row, 0)
        queenside_rook = board.get_piece(queenside_rook_pos)
        if (queenside_rook and 
            queenside_rook.piece_type == PieceType.ROOK and 
            not queenside_rook.has_moved):
            
            # Check if squares between are empty
            if all(board.get_piece(Position(row, c)) is None for c in range(1, 4)):
                # Check if king doesn't move through check
                if not any(board.is_position_under_attack(Position(row, c), self.color) 
                          for c in range(2, 5)):
                    moves.append(Position(row, 2))
        
        return moves


class Queen(Piece):
    """Queen - combines Rook and Bishop movement"""
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.QUEEN
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Queen moves in all 8 directions
        
        Time Complexity: O(n) where n is board size
        """
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        return self._get_linear_moves(board, directions)


class Rook(Piece):
    """Rook - moves horizontally or vertically"""
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.ROOK
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Rook moves in 4 directions (horizontal/vertical)
        
        Time Complexity: O(n) where n is board size
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_linear_moves(board, directions)


class Bishop(Piece):
    """Bishop - moves diagonally"""
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.BISHOP
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Bishop moves in 4 diagonal directions
        
        Time Complexity: O(n) where n is board size
        """
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return self._get_linear_moves(board, directions)


class Knight(Piece):
    """
    Knight - L-shaped moves (2+1 squares)
    
    Interview Focus: Knight is the only piece that can jump over others
    """
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.KNIGHT
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Knight moves in L-shape
        
        Time Complexity: O(1) - always checks 8 positions
        """
        moves = []
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, dc in knight_moves:
            new_pos = Position(self.position.row + dr, self.position.col + dc)
            if not new_pos.is_valid():
                continue
            
            target_piece = board.get_piece(new_pos)
            if target_piece is None or target_piece.color != self.color:
                moves.append(new_pos)
        
        return moves


class Pawn(Piece):
    """
    Pawn - most complex piece due to special rules
    
    Special moves:
    - Double move from starting position
    - En passant capture
    - Promotion when reaching end
    
    Interview Focus: How do you handle direction-dependent and conditional moves?
    """
    
    def __init__(self, color: Color, position: Position):
        super().__init__(color, position)
        self.piece_type = PieceType.PAWN
    
    def get_possible_moves(self, board: 'Board') -> List[Position]:
        """
        Pawn moves forward, captures diagonally
        
        Key Challenge: Direction depends on color, multiple special cases
        Time Complexity: O(1) - checks fixed number of squares
        """
        moves = []
        direction = -1 if self.color == Color.WHITE else 1  # White moves up, black down
        
        # Forward move
        forward_pos = Position(self.position.row + direction, self.position.col)
        if forward_pos.is_valid() and board.get_piece(forward_pos) is None:
            moves.append(forward_pos)
            
            # Double move from starting position
            if not self.has_moved:
                double_pos = Position(self.position.row + 2 * direction, self.position.col)
                if board.get_piece(double_pos) is None:
                    moves.append(double_pos)
        
        # Diagonal captures
        for dc in [-1, 1]:
            capture_pos = Position(self.position.row + direction, self.position.col + dc)
            if not capture_pos.is_valid():
                continue
            
            target_piece = board.get_piece(capture_pos)
            if target_piece and target_piece.color != self.color:
                moves.append(capture_pos)
        
        # En passant
        moves.extend(self._get_en_passant_moves(board))
        
        return moves
    
    def _get_en_passant_moves(self, board: 'Board') -> List[Position]:
        """
        Get en passant capture moves
        
        En passant requirements:
        1. Adjacent pawn just moved two squares
        2. Must be captured immediately after that move
        """
        if not board.last_move:
            return []
        
        moves = []
        last_move = board.last_move
        
        # Check if last move was a pawn double-move
        if (last_move.piece.piece_type == PieceType.PAWN and
            abs(last_move.end.row - last_move.start.row) == 2 and
            last_move.end.row == self.position.row):
            
            # Check if pawns are adjacent
            if abs(last_move.end.col - self.position.col) == 1:
                direction = -1 if self.color == Color.WHITE else 1
                en_passant_pos = Position(
                    self.position.row + direction,
                    last_move.end.col
                )
                moves.append(en_passant_pos)
        
        return moves


# ============================================================================
# SECTION 3: BOARD - Game state representation
# ============================================================================

class Board:
    """
    Chess board representation
    
    Interview Focus: How do you represent a game board efficiently?
    
    Design choices:
    - 2D array: O(1) access by position
    - Piece tracking: Quick lookup of king position
    - Move history: Undo and game analysis
    """
    
    def __init__(self):
        """Initialize empty 8x8 board"""
        self.grid: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self.last_move: Optional[Move] = None
        self.move_history: List[Move] = []
    
    def get_piece(self, position: Position) -> Optional[Piece]:
        """
        Get piece at position
        
        Time Complexity: O(1)
        """
        if not position.is_valid():
            return None
        return self.grid[position.row][position.col]
    
    def set_piece(self, position: Position, piece: Optional[Piece]) -> None:
        """
        Place piece at position
        
        Time Complexity: O(1)
        """
        if position.is_valid():
            self.grid[position.row][position.col] = piece
            if piece:
                piece.position = position
    
    def remove_piece(self, position: Position) -> Optional[Piece]:
        """Remove and return piece at position"""
        piece = self.get_piece(position)
        if piece:
            self.grid[position.row][position.col] = None
        return piece
    
    def move_piece(self, start: Position, end: Position) -> Optional[Piece]:
        """
        Move piece from start to end
        
        Returns captured piece if any
        Time Complexity: O(1)
        """
        piece = self.remove_piece(start)
        captured = self.remove_piece(end)
        if piece:
            self.set_piece(end, piece)
            piece.has_moved = True
        return captured
    
    def find_king(self, color: Color) -> Optional[Position]:
        """
        Find king position for given color
        
        Time Complexity: O(n²) in worst case, but only 64 squares
        
        Optimization: Could cache king positions for O(1) lookup
        """
        for row in range(8):
            for col in range(8):
                piece = self.grid[row][col]
                if piece and piece.piece_type == PieceType.KING and piece.color == color:
                    return Position(row, col)
        return None
    
    def is_position_under_attack(self, position: Position, by_color: Color) -> bool:
        """
        Check if position is under attack by pieces of given color
        
        Interview Focus: Critical for check detection
        Time Complexity: O(n²) - check all opponent pieces
        """
        for row in range(8):
            for col in range(8):
                piece = self.grid[row][col]
                if piece and piece.color == by_color.opposite():
                    # Special case for pawns - they attack diagonally but move forward
                    if piece.piece_type == PieceType.PAWN:
                        direction = -1 if piece.color == Color.WHITE else 1
                        attack_positions = [
                            Position(piece.position.row + direction, piece.position.col - 1),
                            Position(piece.position.row + direction, piece.position.col + 1)
                        ]
                        if position in attack_positions:
                            return True
                    else:
                        if position in piece.get_possible_moves(self):
                            return True
        return False
    
    def setup_initial_position(self) -> None:
        """
        Setup standard chess starting position
        
        Interview Focus: How do you initialize game state?
        """
        # Pawns
        for col in range(8):
            self.set_piece(Position(1, col), Pawn(Color.BLACK, Position(1, col)))
            self.set_piece(Position(6, col), Pawn(Color.WHITE, Position(6, col)))
        
        # Rooks
        self.set_piece(Position(0, 0), Rook(Color.BLACK, Position(0, 0)))
        self.set_piece(Position(0, 7), Rook(Color.BLACK, Position(0, 7)))
        self.set_piece(Position(7, 0), Rook(Color.WHITE, Position(7, 0)))
        self.set_piece(Position(7, 7), Rook(Color.WHITE, Position(7, 7)))
        
        # Knights
        self.set_piece(Position(0, 1), Knight(Color.BLACK, Position(0, 1)))
        self.set_piece(Position(0, 6), Knight(Color.BLACK, Position(0, 6)))
        self.set_piece(Position(7, 1), Knight(Color.WHITE, Position(7, 1)))
        self.set_piece(Position(7, 6), Knight(Color.WHITE, Position(7, 6)))
        
        # Bishops
        self.set_piece(Position(0, 2), Bishop(Color.BLACK, Position(0, 2)))
        self.set_piece(Position(0, 5), Bishop(Color.BLACK, Position(0, 5)))
        self.set_piece(Position(7, 2), Bishop(Color.WHITE, Position(7, 2)))
        self.set_piece(Position(7, 5), Bishop(Color.WHITE, Position(7, 5)))
        
        # Queens
        self.set_piece(Position(0, 3), Queen(Color.BLACK, Position(0, 3)))
        self.set_piece(Position(7, 3), Queen(Color.WHITE, Position(7, 3)))
        
        # Kings
        self.set_piece(Position(0, 4), King(Color.BLACK, Position(0, 4)))
        self.set_piece(Position(7, 4), King(Color.WHITE, Position(7, 4)))
    
    def clone(self) -> 'Board':
        """
        Create deep copy of board
        
        Interview Focus: Why deep copy? For move validation without affecting game state
        """
        return deepcopy(self)
    
    def __str__(self) -> str:
        """Pretty print board"""
        lines = []
        lines.append("  a b c d e f g h")
        for row in range(8):
            line = f"{8-row} "
            for col in range(8):
                piece = self.grid[row][col]
                if piece:
                    line += str(piece) + " "
                else:
                    line += ".  "
            line += f"{8-row}"
            lines.append(line)
        lines.append("  a b c d e f g h")
        return "\n".join(lines)


# ============================================================================
# SECTION 4: GAME CONTROLLER - Main game logic
# ============================================================================

class ChessGame:
    """
    Main chess game controller
    
    Responsibilities:
    - Turn management
    - Move validation (including check)
    - Game state detection (check, checkmate, stalemate)
    - Special move handling
    - Move history tracking
    
    Interview Focus: How do you coordinate complex game rules?
    """
    
    def __init__(self, game_id: str = None):
        self.game_id = game_id or f"game_{int(time.time())}"
        self.board = Board()
        self.board.setup_initial_position()
        self.current_turn = Color.WHITE
        self.game_state = GameState.ACTIVE
        self.move_count = 0
    
    def make_move(self, start_notation: str, end_notation: str, 
                  promotion_type: Optional[PieceType] = None) -> Tuple[bool, str]:
        """
        Make a move using algebraic notation
        
        Interview Focus: How do you validate a complete move?
        
        Returns: (success, message)
        
        Key Challenges:
        - Validate piece can move to target
        - Ensure move doesn't leave king in check
        - Handle special moves (castling, en passant, promotion)
        - Update game state (check, checkmate, stalemate)
        """
        if self.game_state in [GameState.CHECKMATE, GameState.STALEMATE, GameState.DRAW]:
            return False, f"Game is over: {self.game_state.value}"
        
        try:
            start = Position.from_algebraic(start_notation)
            end = Position.from_algebraic(end_notation)
        except (ValueError, IndexError):
            return False, "Invalid position notation"
        
        # Validate start position has a piece
        piece = self.board.get_piece(start)
        if not piece:
            return False, "No piece at start position"
        
        # Validate it's the correct player's turn
        if piece.color != self.current_turn:
            return False, f"It's {self.current_turn.value}'s turn"
        
        # Validate the move is legal for this piece
        if end not in piece.get_possible_moves(self.board):
            return False, f"{piece.piece_type.value} cannot move to {end_notation}"
        
        # Create move object
        captured_piece = self.board.get_piece(end)
        move = Move(
            start=start,
            end=end,
            piece=piece,
            captured_piece=captured_piece,
            promotion_type=promotion_type
        )
        
        # Check for special moves
        if piece.piece_type == PieceType.KING and abs(end.col - start.col) == 2:
            move.is_castling = True
        
        if piece.piece_type == PieceType.PAWN:
            # Check en passant
            if abs(end.col - start.col) == 1 and not captured_piece:
                move.is_en_passant = True
            
            # Check promotion
            if (end.row == 0 and piece.color == Color.WHITE) or \
               (end.row == 7 and piece.color == Color.BLACK):
                if not promotion_type:
                    promotion_type = PieceType.QUEEN  # Default to queen
                move.promotion_type = promotion_type
        
        # Validate move doesn't leave king in check
        if not self._is_legal_move(move):
            return False, "Move would leave king in check"
        
        # Execute the move
        self._execute_move(move)
        
        # Update game state
        self.move_count += 1
        self.current_turn = self.current_turn.opposite()
        self._update_game_state()
        
        # Format response message
        move_notation = move.to_algebraic()
        state_msg = ""
        if self.game_state == GameState.CHECK:
            state_msg = " (Check!)"
        elif self.game_state == GameState.CHECKMATE:
            state_msg = f" (Checkmate! {move.piece.color.value.capitalize()} wins!)"
        elif self.game_state == GameState.STALEMATE:
            state_msg = " (Stalemate - Draw!)"
        
        return True, f"Move: {move_notation}{state_msg}"
    
    def _is_legal_move(self, move: Move) -> bool:
        """
        Check if move is legal (doesn't leave king in check)
        
        Interview Focus: Why simulate move? Can't modify actual game state for validation
        Time Complexity: O(n²) - need to check all opponent pieces
        """
        # Clone board and execute move
        test_board = self.board.clone()
        test_piece = test_board.get_piece(move.start)
        
        # Execute move on test board
        if move.is_castling:
            # Move king
            test_board.move_piece(move.start, move.end)
            
            # Move rook
            if move.end.col > move.start.col:  # Kingside
                rook_start = Position(move.start.row, 7)
                rook_end = Position(move.start.row, 5)
            else:  # Queenside
                rook_start = Position(move.start.row, 0)
                rook_end = Position(move.start.row, 3)
            test_board.move_piece(rook_start, rook_end)
        
        elif move.is_en_passant:
            test_board.move_piece(move.start, move.end)
            # Remove captured pawn
            captured_pawn_pos = Position(move.start.row, move.end.col)
            test_board.remove_piece(captured_pawn_pos)
        
        else:
            test_board.move_piece(move.start, move.end)
        
        # Check if king is in check after move
        king_pos = test_board.find_king(move.piece.color)
        if not king_pos:
            return False
        
        return not test_board.is_position_under_attack(king_pos, move.piece.color)
    
    def _execute_move(self, move: Move) -> None:
        """
        Execute a validated move on the board
        
        Interview Focus: How do you handle special move execution?
        """
        if move.is_castling:
            # Move king
            self.board.move_piece(move.start, move.end)
            
            # Move rook
            if move.end.col > move.start.col:  # Kingside
                rook_start = Position(move.start.row, 7)
                rook_end = Position(move.start.row, 5)
            else:  # Queenside
                rook_start = Position(move.start.row, 0)
                rook_end = Position(move.start.row, 3)
            self.board.move_piece(rook_start, rook_end)
        
        elif move.is_en_passant:
            self.board.move_piece(move.start, move.end)
            # Remove captured pawn
            captured_pawn_pos = Position(move.start.row, move.end.col)
            move.captured_piece = self.board.remove_piece(captured_pawn_pos)
        
        else:
            self.board.move_piece(move.start, move.end)
        
        # Handle pawn promotion
        if move.promotion_type:
            piece_classes = {
                PieceType.QUEEN: Queen,
                PieceType.ROOK: Rook,
                PieceType.BISHOP: Bishop,
                PieceType.KNIGHT: Knight
            }
            new_piece = piece_classes[move.promotion_type](move.piece.color, move.end)
            new_piece.has_moved = True
            self.board.set_piece(move.end, new_piece)
        
        # Update board state
        self.board.last_move = move
        self.board.move_history.append(move)
    
    def _update_game_state(self) -> None:
        """
        Update game state (check, checkmate, stalemate)
        
        Interview Focus: How do you detect game-ending conditions?
        
        Key Insight: Checkmate vs Stalemate
        - Checkmate: King in check + no legal moves
        - Stalemate: King NOT in check + no legal moves
        """
        king_pos = self.board.find_king(self.current_turn)
        if not king_pos:
            self.game_state = GameState.CHECKMATE
            return
        
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
    
    def _has_legal_moves(self) -> bool:
        """
        Check if current player has any legal moves
        
        Time Complexity: O(n³) worst case
        - O(n²) to iterate all pieces
        - O(n) to get moves per piece
        - O(n²) to validate each move
        
        Optimization: Could cache or short-circuit on first legal move
        """
        for row in range(8):
            for col in range(8):
                piece = self.board.grid[row][col]
                if not piece or piece.color != self.current_turn:
                    continue
                
                for target_pos in piece.get_possible_moves(self.board):
                    move = Move(
                        start=piece.position,
                        end=target_pos,
                        piece=piece,
                        captured_piece=self.board.get_piece(target_pos)
                    )
                    if self._is_legal_move(move):
                        return True  # Short-circuit on first legal move
        
        return False
    
    def get_valid_moves(self, position_notation: str) -> List[str]:
        """
        Get all valid moves for piece at position
        
        Interview Focus: How do you show available moves to player?
        """
        try:
            position = Position.from_algebraic(position_notation)
        except (ValueError, IndexError):
            return []
        
        piece = self.board.get_piece(position)
        if not piece or piece.color != self.current_turn:
            return []
        
        valid_moves = []
        for target_pos in piece.get_possible_moves(self.board):
            move = Move(
                start=position,
                end=target_pos,
                piece=piece,
                captured_piece=self.board.get_piece(target_pos)
            )
            if self._is_legal_move(move):
                valid_moves.append(target_pos.to_algebraic())
        
        return valid_moves
    
    def get_game_status(self) -> Dict:
        """Get current game status"""
        return {
            "game_id": self.game_id,
            "current_turn": self.current_turn.value,
            "game_state": self.game_state.value,
            "move_count": self.move_count,
            "moves_played": len(self.board.move_history)
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


def demo_basic_moves():
    """Demonstrate basic piece movements"""
    print_separator("Basic Piece Movements")
    
    game = ChessGame("demo1")
    print("\nInitial board position:")
    print(game.board)
    
    # Scholar's Mate sequence
    moves = [
        ("e2", "e4", "1. e4 - Pawn to e4"),
        ("e7", "e5", "1... e5 - Black responds"),
        ("f1", "c4", "2. Bc4 - Bishop develops"),
        ("b8", "c6", "2... Nc6 - Knight develops"),
        ("d1", "h5", "3. Qh5 - Queen attacks"),
        ("g8", "f6", "3... Nf6 - Knight defends"),
        ("h5", "f7", "4. Qxf7# - Checkmate!"),
    ]
    
    print("\n🔹 Demonstrating Scholar's Mate:")
    for start, end, description in moves:
        print(f"\n{description}")
        success, message = game.make_move(start, end)
        if success:
            print(f"✓ {message}")
            print(game.board)
            if "Checkmate" in message:
                print("\n🎯 Game Over! White wins by Scholar's Mate!")
                break
        else:
            print(f"✗ {message}")


def demo_special_moves():
    """Demonstrate castling, en passant, and promotion"""
    print_separator("Special Moves")
    
    # Castling demonstration
    print("\n🔹 Demonstrating Castling:")
    game = ChessGame("castling_demo")
    
    # Setup for castling
    castling_setup = [
        ("e2", "e4"),
        ("e7", "e5"),
        ("g1", "f3"),
        ("b8", "c6"),
        ("f1", "c4"),
        ("g8", "f6"),
    ]
    
    for start, end in castling_setup:
        game.make_move(start, end)
    
    print("\nBoard before castling:")
    print(game.board)
    
    success, message = game.make_move("e1", "g1")  # Kingside castling
    print(f"\n{message}")
    print("\nBoard after White castles kingside (O-O):")
    print(game.board)
    
    # En Passant demonstration
    print("\n\n🔹 Demonstrating En Passant:")
    game2 = ChessGame("en_passant_demo")
    
    en_passant_setup = [
        ("e2", "e4"),
        ("a7", "a6"),
        ("e4", "e5"),
        ("d7", "d5"),  # Black pawn moves two squares
    ]
    
    for start, end in en_passant_setup:
        game2.make_move(start, end)
    
    print("\nBoard setup - Black pawn just moved two squares:")
    print(game2.board)
    
    success, message = game2.make_move("e5", "d6")  # En passant capture
    print(f"\n{message}")
    print("\nAfter en passant capture:")
    print(game2.board)
    
    # Pawn Promotion demonstration
    print("\n\n🔹 Demonstrating Pawn Promotion:")
    game3 = ChessGame("promotion_demo")
    
    # Clear board and setup promotion scenario
    game3.board = Board()
    game3.board.set_piece(Position(1, 4), Pawn(Color.WHITE, Position(1, 4)))
    game3.board.set_piece(Position(0, 4), King(Color.BLACK, Position(0, 4)))
    game3.board.set_piece(Position(7, 0), King(Color.WHITE, Position(7, 0)))
    
    print("\nSetup - White pawn about to promote:")
    print(game3.board)
    
    success, message = game3.make_move("e7", "e8", PieceType.QUEEN)
    print(f"\n{message}")
    print("\nAfter pawn promotes to Queen:")
    print(game3.board)


def demo_check_and_checkmate():
    """Demonstrate check and checkmate scenarios"""
    print_separator("Check and Checkmate Detection")
    
    # Fool's Mate - fastest checkmate
    print("\n🔹 Demonstrating Fool's Mate (fastest checkmate):")
    game = ChessGame("fools_mate")
    
    moves = [
        ("f2", "f3", "1. f3 (weak move)"),
        ("e7", "e5", "1... e5"),
        ("g2", "g4", "2. g4 (disaster)"),
        ("d8", "h4", "2... Qh4# (checkmate!)"),
    ]
    
    for start, end, description in moves:
        print(f"\n{description}")
        success, message = game.make_move(start, end)
        print(f"✓ {message}")
        print(game.board)
        
        if "Checkmate" in message:
            print("\n🎯 Fool's Mate! Black wins in 2 moves!")
            break
    
    # Stalemate demonstration
    print("\n\n🔹 Demonstrating Stalemate:")
    game2 = ChessGame("stalemate_demo")
    game2.board = Board()
    
    # Setup stalemate position
    game2.board.set_piece(Position(0, 7), King(Color.BLACK, Position(0, 7)))
    game2.board.set_piece(Position(2, 6), Queen(Color.WHITE, Position(2, 6)))
    game2.board.set_piece(Position(7, 0), King(Color.WHITE, Position(7, 0)))
    game2.current_turn = Color.WHITE
    
    print("\nSetup - About to create stalemate:")
    print(game2.board)
    
    success, message = game2.make_move("g6", "g7")
    print(f"\n{message}")
    print(game2.board)
    
    if "Stalemate" in message:
        print("\n⚖️ Stalemate! Black king not in check but has no legal moves!")


def demo_move_validation():
    """Demonstrate move validation and error handling"""
    print_separator("Move Validation")
    
    game = ChessGame("validation_demo")
    print("\nInitial position:")
    print(game.board)
    
    print("\n🔹 Testing various invalid moves:")
    
    # Test invalid moves
    test_cases = [
        ("e2", "e5", "Pawn can't move 3 squares"),
        ("a2", "a3", "Valid pawn move"),
        ("a3", "a5", "Pawn can't move 2 squares after first move"),
        ("b1", "d2", "Valid knight move"),
        ("d2", "e4", "Valid knight move"),
        ("e1", "e2", "King can't move - square occupied"),
        ("f1", "a6", "Bishop blocked by pawn"),
    ]
    
    for start, end, description in test_cases:
        print(f"\nTrying: {start} to {end} - {description}")
        success, message = game.make_move(start, end)
        print(f"  {'✓' if success else '✗'} {message}")
    
    print("\n\n🔹 Testing move validity check:")
    print("\nValid moves for white pawn at e2:")
    valid_moves = game.get_valid_moves("e2")
    print(f"  {', '.join(valid_moves)}")
    
    print("\nValid moves for white knight at g1:")
    valid_moves = game.get_valid_moves("g1")
    print(f"  {', '.join(valid_moves)}")


def demo_game_flow():
    """Demonstrate complete game flow"""
    print_separator("Complete Game Flow")
    
    game = ChessGame("game_flow_demo")
    
    print("\n🔹 Playing a short game:")
    print(game.board)
    
    # Opening moves
    moves = [
        ("e2", "e4"),
        ("e7", "e5"),
        ("g1", "f3"),
        ("b8", "c6"),
        ("f1", "b5"),  # Spanish Opening
        ("a7", "a6"),
        ("b5", "a4"),
        ("g8", "f6"),
    ]
    
    for i, (start, end) in enumerate(moves, 1):
        success, message = game.make_move(start, end)
        if success:
            move_num = (i + 1) // 2
            color = "White" if i % 2 == 1 else "Black"
            print(f"\nMove {move_num} ({color}): {message}")
            
            status = game.get_game_status()
            print(f"  State: {status['game_state']}, Turn: {status['current_turn']}")
    
    print("\n\nFinal board position:")
    print(game.board)
    
    print("\n\nGame statistics:")
    status = game.get_game_status()
    for key, value in status.items():
        print(f"  {key}: {value}")


def run_demo():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("  CHESS GAME - COMPREHENSIVE DEMONSTRATION")
    print("  Features: Complete chess rules with special moves")
    print("="*70)
    
    demo_basic_moves()
    demo_special_moves()
    demo_check_and_checkmate()
    demo_move_validation()
    demo_game_flow()
    
    print_separator()
    print("✅ All demonstrations completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Main entry point for demonstration
    
    Usage:
        python 09_chess_game.py
    """
    run_demo()
