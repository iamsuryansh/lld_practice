"""
Library Management System - Interview-Grade Implementation

Key Design Patterns Demonstrated:
- Strategy Pattern: Fine calculation strategies
- Observer Pattern: Notification system
- State Pattern: Book status management
- Command Pattern: Transaction operations
- Factory Pattern: User type creation

Focus Areas for Interviews:
- Queue management for reservations
- Fine calculation algorithms
- Notification systems
- Concurrent borrowing/returning
- Search and cataloging strategies
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import uuid


class BookStatus(Enum):
    AVAILABLE = "available"
    BORROWED = "borrowed"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    LOST = "lost"


class UserType(Enum):
    STUDENT = "student"
    FACULTY = "faculty"
    STAFF = "staff"
    GUEST = "guest"


class NotificationType(Enum):
    DUE_DATE_REMINDER = "due_date_reminder"
    OVERDUE_NOTICE = "overdue_notice"
    BOOK_AVAILABLE = "book_available"
    RESERVATION_CONFIRMED = "reservation_confirmed"
    FINE_NOTICE = "fine_notice"


@dataclass
class Book:
    """
    Book entity - Core domain object
    Interview Focus: How do you model book uniqueness vs copies?
    """
    isbn: str
    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    category: str
    location: str
    copy_id: str = None
    status: BookStatus = BookStatus.AVAILABLE
    
    def __post_init__(self):
        if not self.copy_id:
            self.copy_id = f"{self.isbn}_{uuid.uuid4().hex[:8]}"


@dataclass
class User:
    """
    User entity with borrowing limits based on type
    Interview Focus: How do you handle different user privileges?
    """
    user_id: str
    name: str
    email: str
    phone: str
    user_type: UserType
    registration_date: datetime
    is_active: bool = True
    
    @property
    def max_books_allowed(self) -> int:
        """Strategy pattern for user-type specific limits"""
        limits = {
            UserType.STUDENT: 5,
            UserType.FACULTY: 15,
            UserType.STAFF: 10,
            UserType.GUEST: 2
        }
        return limits.get(self.user_type, 2)
    
    @property
    def loan_period_days(self) -> int:
        """Different loan periods for different user types"""
        periods = {
            UserType.STUDENT: 14,
            UserType.FACULTY: 30,
            UserType.STAFF: 21,
            UserType.GUEST: 7
        }
        return periods.get(self.user_type, 7)


@dataclass
class Transaction:
    """
    Transaction record for audit trail
    Interview Focus: How do you maintain transaction history?
    """
    transaction_id: str
    user_id: str
    book_copy_id: str
    transaction_type: str  # BORROW, RETURN, RESERVE, CANCEL_RESERVATION
    timestamp: datetime
    due_date: Optional[datetime] = None
    return_date: Optional[datetime] = None
    fine_amount: float = 0.0
    notes: str = ""


class FineCalculationStrategy(ABC):
    """
    Strategy pattern for different fine calculation methods
    Interview Focus: How do you handle changing business rules?
    """
    
    @abstractmethod
    def calculate_fine(self, days_overdue: int, user_type: UserType, book_category: str) -> float:
        pass


class StandardFineCalculator(FineCalculationStrategy):
    """Standard fine calculation - flat rate per day"""
    
    def __init__(self):
        self.base_rate_per_day = 0.50
        self.user_type_multipliers = {
            UserType.STUDENT: 1.0,
            UserType.FACULTY: 0.5,
            UserType.STAFF: 0.75,
            UserType.GUEST: 1.5
        }
    
    def calculate_fine(self, days_overdue: int, user_type: UserType, book_category: str) -> float:
        if days_overdue <= 0:
            return 0.0
        
        base_fine = days_overdue * self.base_rate_per_day
        multiplier = self.user_type_multipliers.get(user_type, 1.0)
        
        # Higher fine for reference books
        if book_category.lower() in ['reference', 'rare']:
            multiplier *= 2.0
        
        return round(base_fine * multiplier, 2)


class ProgressiveFineCalculator(FineCalculationStrategy):
    """Progressive fine calculation - increases over time"""
    
    def calculate_fine(self, days_overdue: int, user_type: UserType, book_category: str) -> float:
        if days_overdue <= 0:
            return 0.0
        
        # Progressive rates: first week 0.50, second week 1.00, after that 2.00
        fine = 0.0
        remaining_days = days_overdue
        
        # First 7 days
        if remaining_days > 0:
            days_in_tier = min(remaining_days, 7)
            fine += days_in_tier * 0.50
            remaining_days -= days_in_tier
        
        # Next 7 days
        if remaining_days > 0:
            days_in_tier = min(remaining_days, 7)
            fine += days_in_tier * 1.00
            remaining_days -= days_in_tier
        
        # Remaining days
        if remaining_days > 0:
            fine += remaining_days * 2.00
        
        return round(fine, 2)


class NotificationObserver(ABC):
    """
    Observer pattern for notifications
    Interview Focus: How do you decouple notification logic?
    """
    
    @abstractmethod
    def notify(self, user: User, notification_type: NotificationType, message: str, **kwargs):
        pass


class EmailNotificationObserver(NotificationObserver):
    """Email notification implementation"""
    
    def notify(self, user: User, notification_type: NotificationType, message: str, **kwargs):
        print(f"📧 EMAIL to {user.email}: [{notification_type.value.upper()}] {message}")


class SMSNotificationObserver(NotificationObserver):
    """SMS notification implementation"""
    
    def notify(self, user: User, notification_type: NotificationType, message: str, **kwargs):
        print(f"📱 SMS to {user.phone}: [{notification_type.value.upper()}] {message}")


class InAppNotificationObserver(NotificationObserver):
    """In-app notification implementation"""
    
    def __init__(self):
        self.notifications: Dict[str, List[dict]] = defaultdict(list)
    
    def notify(self, user: User, notification_type: NotificationType, message: str, **kwargs):
        notification = {
            'type': notification_type.value,
            'message': message,
            'timestamp': datetime.now(),
            'read': False
        }
        self.notifications[user.user_id].append(notification)
        print(f"🔔 IN-APP for {user.name}: {message}")


class ReservationQueue:
    """
    Queue management for book reservations
    Interview Focus: How do you handle fair queuing and priority?
    """
    
    def __init__(self):
        self.queues: Dict[str, deque] = defaultdict(deque)  # isbn -> queue of user_ids
        self.user_reservations: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of isbns
        self.lock = threading.RLock()
    
    def add_reservation(self, isbn: str, user_id: str) -> int:
        """Add user to reservation queue. Returns position in queue."""
        with self.lock:
            if user_id in self.queues[isbn]:
                raise ValueError(f"User {user_id} already has reservation for {isbn}")
            
            self.queues[isbn].append(user_id)
            self.user_reservations[user_id].add(isbn)
            return len(self.queues[isbn])
    
    def get_next_user(self, isbn: str) -> Optional[str]:
        """Get next user in reservation queue"""
        with self.lock:
            if not self.queues[isbn]:
                return None
            
            user_id = self.queues[isbn].popleft()
            self.user_reservations[user_id].discard(isbn)
            return user_id
    
    def remove_reservation(self, isbn: str, user_id: str) -> bool:
        """Remove user's reservation"""
        with self.lock:
            if user_id not in self.queues[isbn]:
                return False
            
            # Convert to list, remove, convert back to deque
            queue_list = list(self.queues[isbn])
            queue_list.remove(user_id)
            self.queues[isbn] = deque(queue_list)
            self.user_reservations[user_id].discard(isbn)
            return True
    
    def get_user_position(self, isbn: str, user_id: str) -> Optional[int]:
        """Get user's position in reservation queue"""
        with self.lock:
            try:
                return list(self.queues[isbn]).index(user_id) + 1
            except ValueError:
                return None
    
    def get_user_reservations(self, user_id: str) -> Set[str]:
        """Get all books reserved by user"""
        return self.user_reservations[user_id].copy()


class SearchEngine:
    """
    Book search and cataloging system
    Interview Focus: How do you handle complex search queries?
    """
    
    def __init__(self):
        self.title_index: Dict[str, Set[str]] = defaultdict(set)  # word -> set of copy_ids
        self.author_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.isbn_index: Dict[str, Set[str]] = defaultdict(set)
    
    def index_book(self, book: Book):
        """Add book to search indices"""
        # Index title words
        for word in book.title.lower().split():
            self.title_index[word].add(book.copy_id)
        
        # Index authors
        for author in book.authors:
            for word in author.lower().split():
                self.author_index[word].add(book.copy_id)
        
        # Index category
        self.category_index[book.category.lower()].add(book.copy_id)
        
        # Index ISBN
        self.isbn_index[book.isbn].add(book.copy_id)
    
    def remove_book(self, book: Book):
        """Remove book from search indices"""
        # Remove from title index
        for word in book.title.lower().split():
            self.title_index[word].discard(book.copy_id)
        
        # Remove from author index
        for author in book.authors:
            for word in author.lower().split():
                self.author_index[word].discard(book.copy_id)
        
        # Remove from other indices
        self.category_index[book.category.lower()].discard(book.copy_id)
        self.isbn_index[book.isbn].discard(book.copy_id)
    
    def search(self, query: str, search_type: str = "all") -> Set[str]:
        """
        Search books by query
        search_type: 'title', 'author', 'category', 'isbn', 'all'
        """
        query_words = query.lower().split()
        result_sets = []
        
        if search_type in ['title', 'all']:
            for word in query_words:
                if word in self.title_index:
                    result_sets.append(self.title_index[word])
        
        if search_type in ['author', 'all']:
            for word in query_words:
                if word in self.author_index:
                    result_sets.append(self.author_index[word])
        
        if search_type in ['category', 'all']:
            for word in query_words:
                if word in self.category_index:
                    result_sets.append(self.category_index[word])
        
        if search_type in ['isbn', 'all']:
            for word in query_words:
                if word in self.isbn_index:
                    result_sets.append(self.isbn_index[word])
        
        if not result_sets:
            return set()
        
        # Intersection of all result sets
        result = result_sets[0]
        for result_set in result_sets[1:]:
            result = result.intersection(result_set)
        
        return result


class LibraryManagementSystem:
    """
    Main library management system
    Interview Focus: How do you coordinate multiple subsystems?
    """
    
    def __init__(self):
        # Data storage
        self.books: Dict[str, Book] = {}  # copy_id -> Book
        self.users: Dict[str, User] = {}  # user_id -> User
        self.transactions: List[Transaction] = []
        
        # Tracking
        self.borrowed_books: Dict[str, str] = {}  # copy_id -> user_id
        self.user_borrowed_books: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of copy_ids
        self.user_fines: Dict[str, float] = defaultdict(float)
        
        # Components
        self.reservation_queue = ReservationQueue()
        self.search_engine = SearchEngine()
        self.fine_calculator: FineCalculationStrategy = StandardFineCalculator()
        
        # Observers
        self.notification_observers: List[NotificationObserver] = []
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_notification_observer(self, observer: NotificationObserver):
        """Add notification observer"""
        self.notification_observers.append(observer)
    
    def notify_observers(self, user: User, notification_type: NotificationType, message: str, **kwargs):
        """Notify all observers"""
        for observer in self.notification_observers:
            observer.notify(user, notification_type, message, **kwargs)
    
    def add_book(self, book: Book) -> str:
        """Add book to library catalog"""
        with self.lock:
            self.books[book.copy_id] = book
            self.search_engine.index_book(book)
            return book.copy_id
    
    def add_user(self, user: User) -> str:
        """Add user to library system"""
        with self.lock:
            self.users[user.user_id] = user
            return user.user_id
    
    def search_books(self, query: str, search_type: str = "all") -> List[Book]:
        """Search for books"""
        copy_ids = self.search_engine.search(query, search_type)
        return [self.books[copy_id] for copy_id in copy_ids if copy_id in self.books]
    
    def borrow_book(self, user_id: str, copy_id: str) -> Transaction:
        """
        Borrow a book
        Interview Focus: How do you handle concurrent borrows?
        """
        with self.lock:
            # Validate user
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            user = self.users[user_id]
            if not user.is_active:
                raise ValueError(f"User {user_id} is inactive")
            
            # Check borrowing limits
            current_borrowed = len(self.user_borrowed_books[user_id])
            if current_borrowed >= user.max_books_allowed:
                raise ValueError(f"User has reached maximum borrowing limit ({user.max_books_allowed})")
            
            # Check for outstanding fines
            if self.user_fines[user_id] > 50.00:  # $50 fine limit
                raise ValueError(f"User has outstanding fines: ${self.user_fines[user_id]:.2f}")
            
            # Validate book
            if copy_id not in self.books:
                raise ValueError(f"Book copy {copy_id} not found")
            
            book = self.books[copy_id]
            
            if book.status != BookStatus.AVAILABLE:
                # Check if user has reservation
                if book.status == BookStatus.RESERVED:
                    next_user = self.reservation_queue.get_next_user(book.isbn)
                    if next_user != user_id:
                        raise ValueError(f"Book is reserved for another user")
                else:
                    raise ValueError(f"Book is not available (status: {book.status.value})")
            
            # Process borrowing
            transaction_id = str(uuid.uuid4())
            borrow_date = datetime.now()
            due_date = borrow_date + timedelta(days=user.loan_period_days)
            
            # Update book status
            book.status = BookStatus.BORROWED
            
            # Update tracking
            self.borrowed_books[copy_id] = user_id
            self.user_borrowed_books[user_id].add(copy_id)
            
            # Create transaction
            transaction = Transaction(
                transaction_id=transaction_id,
                user_id=user_id,
                book_copy_id=copy_id,
                transaction_type="BORROW",
                timestamp=borrow_date,
                due_date=due_date
            )
            self.transactions.append(transaction)
            
            # Send notification
            message = f"Book '{book.title}' borrowed successfully. Due date: {due_date.strftime('%Y-%m-%d')}"
            self.notify_observers(user, NotificationType.DUE_DATE_REMINDER, message)
            
            return transaction
    
    def return_book(self, user_id: str, copy_id: str) -> Transaction:
        """
        Return a book
        Interview Focus: How do you handle fine calculations and notifications?
        """
        with self.lock:
            # Validate
            if copy_id not in self.borrowed_books:
                raise ValueError(f"Book {copy_id} is not currently borrowed")
            
            if self.borrowed_books[copy_id] != user_id:
                raise ValueError(f"Book {copy_id} is not borrowed by user {user_id}")
            
            user = self.users[user_id]
            book = self.books[copy_id]
            
            # Find original borrow transaction
            borrow_transaction = None
            for txn in reversed(self.transactions):
                if (txn.book_copy_id == copy_id and 
                    txn.user_id == user_id and 
                    txn.transaction_type == "BORROW" and 
                    txn.return_date is None):
                    borrow_transaction = txn
                    break
            
            if not borrow_transaction:
                raise ValueError("Original borrow transaction not found")
            
            return_date = datetime.now()
            
            # Calculate fine
            fine_amount = 0.0
            if return_date > borrow_transaction.due_date:
                days_overdue = (return_date - borrow_transaction.due_date).days
                fine_amount = self.fine_calculator.calculate_fine(
                    days_overdue, user.user_type, book.category
                )
                self.user_fines[user_id] += fine_amount
            
            # Update book status
            book.status = BookStatus.AVAILABLE
            
            # Update tracking
            del self.borrowed_books[copy_id]
            self.user_borrowed_books[user_id].discard(copy_id)
            
            # Update original transaction
            borrow_transaction.return_date = return_date
            borrow_transaction.fine_amount = fine_amount
            
            # Create return transaction
            transaction_id = str(uuid.uuid4())
            return_transaction = Transaction(
                transaction_id=transaction_id,
                user_id=user_id,
                book_copy_id=copy_id,
                transaction_type="RETURN",
                timestamp=return_date,
                fine_amount=fine_amount
            )
            self.transactions.append(return_transaction)
            
            # Check for reservations
            next_user_id = self.reservation_queue.get_next_user(book.isbn)
            if next_user_id and next_user_id in self.users:
                book.status = BookStatus.RESERVED
                next_user = self.users[next_user_id]
                message = f"Book '{book.title}' is now available for pickup. You have 24 hours to borrow."
                self.notify_observers(next_user, NotificationType.BOOK_AVAILABLE, message)
            
            # Send fine notification if applicable
            if fine_amount > 0:
                message = f"Fine of ${fine_amount:.2f} applied for late return of '{book.title}'"
                self.notify_observers(user, NotificationType.FINE_NOTICE, message)
            
            return return_transaction
    
    def reserve_book(self, user_id: str, isbn: str) -> int:
        """
        Reserve a book
        Interview Focus: How do you handle reservation queuing?
        """
        with self.lock:
            # Validate user
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            user = self.users[user_id]
            
            # Check if any copy is available
            available_copy = None
            for copy_id, book in self.books.items():
                if book.isbn == isbn and book.status == BookStatus.AVAILABLE:
                    available_copy = book
                    break
            
            if available_copy:
                raise ValueError("Book is currently available for borrowing")
            
            # Check if user already has this book borrowed
            for copy_id in self.user_borrowed_books[user_id]:
                if self.books[copy_id].isbn == isbn:
                    raise ValueError("User already has this book borrowed")
            
            # Add to reservation queue
            position = self.reservation_queue.add_reservation(isbn, user_id)
            
            # Create reservation transaction
            transaction_id = str(uuid.uuid4())
            transaction = Transaction(
                transaction_id=transaction_id,
                user_id=user_id,
                book_copy_id=isbn,  # Using ISBN for reservation
                transaction_type="RESERVE",
                timestamp=datetime.now()
            )
            self.transactions.append(transaction)
            
            # Send notification
            book_title = next((book.title for book in self.books.values() 
                             if book.isbn == isbn), "Unknown")
            message = f"Reservation confirmed for '{book_title}'. Position in queue: {position}"
            self.notify_observers(user, NotificationType.RESERVATION_CONFIRMED, message)
            
            return position
    
    def get_overdue_books(self) -> List[dict]:
        """Get list of overdue books"""
        overdue_books = []
        current_time = datetime.now()
        
        for txn in self.transactions:
            if (txn.transaction_type == "BORROW" and 
                txn.return_date is None and 
                txn.due_date < current_time):
                
                days_overdue = (current_time - txn.due_date).days
                book = self.books[txn.book_copy_id]
                user = self.users[txn.user_id]
                
                overdue_books.append({
                    'transaction': txn,
                    'book': book,
                    'user': user,
                    'days_overdue': days_overdue
                })
        
        return overdue_books
    
    def send_overdue_notifications(self):
        """Send notifications for overdue books"""
        overdue_books = self.get_overdue_books()
        
        for overdue_info in overdue_books:
            user = overdue_info['user']
            book = overdue_info['book']
            days_overdue = overdue_info['days_overdue']
            
            message = f"Book '{book.title}' is {days_overdue} days overdue. Please return immediately."
            self.notify_observers(user, NotificationType.OVERDUE_NOTICE, message)
    
    def pay_fine(self, user_id: str, amount: float) -> bool:
        """Pay user fine"""
        with self.lock:
            if user_id not in self.users:
                return False
            
            if amount <= 0:
                return False
            
            current_fine = self.user_fines[user_id]
            if amount > current_fine:
                amount = current_fine
            
            self.user_fines[user_id] -= amount
            
            # Create payment transaction
            transaction_id = str(uuid.uuid4())
            transaction = Transaction(
                transaction_id=transaction_id,
                user_id=user_id,
                book_copy_id="",
                transaction_type="FINE_PAYMENT",
                timestamp=datetime.now(),
                fine_amount=-amount,  # Negative for payment
                notes=f"Fine payment: ${amount:.2f}"
            )
            self.transactions.append(transaction)
            
            return True
    
    def get_user_summary(self, user_id: str) -> dict:
        """Get comprehensive user summary"""
        if user_id not in self.users:
            return {}
        
        user = self.users[user_id]
        borrowed_books = [self.books[copy_id] for copy_id in self.user_borrowed_books[user_id]]
        reservations = self.reservation_queue.get_user_reservations(user_id)
        
        return {
            'user': user,
            'borrowed_books': borrowed_books,
            'reservations': list(reservations),
            'outstanding_fine': self.user_fines[user_id],
            'transaction_history': [txn for txn in self.transactions if txn.user_id == user_id]
        }


# Demo Usage
def demo_library_system():
    """
    Comprehensive demo showing key features
    Interview Focus: How do you test complex systems?
    """
    print("🏛️ Library Management System Demo")
    print("=" * 50)
    
    # Initialize system
    library = LibraryManagementSystem()
    
    # Add notification observers
    library.add_notification_observer(EmailNotificationObserver())
    library.add_notification_observer(SMSNotificationObserver())
    library.add_notification_observer(InAppNotificationObserver())
    
    # Add users
    student = User(
        user_id="u001",
        name="Alice Johnson",
        email="alice@university.edu",
        phone="+1-555-0101",
        user_type=UserType.STUDENT,
        registration_date=datetime.now()
    )
    
    faculty = User(
        user_id="u002",
        name="Dr. Bob Smith",
        email="bob@university.edu",
        phone="+1-555-0102",
        user_type=UserType.FACULTY,
        registration_date=datetime.now()
    )
    
    library.add_user(student)
    library.add_user(faculty)
    
    # Add books
    book1 = Book(
        isbn="978-0134685991",
        title="Effective Java",
        authors=["Joshua Bloch"],
        publisher="Addison-Wesley",
        publication_year=2017,
        category="Computer Science"
    )
    
    book2 = Book(
        isbn="978-0134685991",  # Same ISBN, different copy
        title="Effective Java",
        authors=["Joshua Bloch"],
        publisher="Addison-Wesley",
        publication_year=2017,
        category="Computer Science"
    )
    
    book3 = Book(
        isbn="978-0132350884",
        title="Clean Code",
        authors=["Robert Martin"],
        publisher="Prentice Hall",
        publication_year=2008,
        category="Computer Science"
    )
    
    library.add_book(book1)
    library.add_book(book2)
    library.add_book(book3)
    
    print("\n📚 Books added to catalog")
    
    # Search books
    print("\n🔍 Search Results for 'Java':")
    search_results = library.search_books("Java")
    for book in search_results:
        print(f"  - {book.title} by {', '.join(book.authors)} [{book.copy_id}]")
    
    # Borrow books
    print("\n📖 Borrowing books...")
    try:
        txn1 = library.borrow_book("u001", book1.copy_id)
        print(f"✅ Alice borrowed Effective Java (copy 1)")
        
        txn2 = library.borrow_book("u002", book2.copy_id)
        print(f"✅ Dr. Smith borrowed Effective Java (copy 2)")
        
        # Try to reserve when copies are available
        print("\n🔒 Attempting reservation...")
        try:
            library.reserve_book("u001", book3.isbn)
        except ValueError as e:
            print(f"❌ Reservation failed: {e}")
        
        # Borrow the available book instead
        txn3 = library.borrow_book("u001", book3.copy_id)
        print(f"✅ Alice borrowed Clean Code")
        
    except Exception as e:
        print(f"❌ Borrowing failed: {e}")
    
    # Try to borrow when all copies are borrowed
    print("\n🔒 Reserving book when all copies borrowed...")
    try:
        position = library.reserve_book("u002", book1.isbn)  # Effective Java
        print(f"✅ Dr. Smith reserved Effective Java, position: {position}")
    except Exception as e:
        print(f"❌ Reservation failed: {e}")
    
    # Return book and trigger reservation notification
    print("\n📤 Returning book...")
    try:
        library.return_book("u001", book1.copy_id)
        print(f"✅ Alice returned Effective Java (copy 1)")
    except Exception as e:
        print(f"❌ Return failed: {e}")
    
    # Simulate overdue scenario
    print("\n⏰ Simulating overdue scenario...")
    # Modify due date to make book overdue
    for txn in library.transactions:
        if txn.transaction_type == "BORROW" and txn.return_date is None:
            txn.due_date = datetime.now() - timedelta(days=5)
    
    library.send_overdue_notifications()
    
    # User summary
    print("\n👤 User Summary for Alice:")
    summary = library.get_user_summary("u001")
    print(f"  Borrowed books: {len(summary['borrowed_books'])}")
    print(f"  Reservations: {len(summary['reservations'])}")
    print(f"  Outstanding fine: ${summary['outstanding_fine']:.2f}")
    
    print("\n✨ Demo completed!")


if __name__ == "__main__":
    demo_library_system()