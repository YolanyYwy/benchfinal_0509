"""Tools for the instore domain."""

from datetime import datetime
from typing import Dict, List, Optional
import hashlib

from AGentCL.domains.instore.data_model import (
    InstoreDB, Order, Shop, ShopProduct, BookInfo, ReservationInfo, Location
)
from AGentCL.environment.toolkit import ToolKitBase, ToolType, is_tool


class InstoreTools(ToolKitBase):
    """Tools for the instore domain."""

    db: InstoreDB

    def __init__(self, db: InstoreDB):
        super().__init__(db)

    def _check_user(self, user_id: str) -> bool:
        """Check if user_id is valid (placeholder - always returns True)."""
        return True

    def _get_shop_tags(self) -> Dict[str, str]:
        """Get shop tags for search."""
        tag_dict = {}
        for shop in self.db.shops.values():
            tag_dict[shop.shop_id] = shop.shop_name + ',' + ','.join(shop.tags)
        return tag_dict

    def _get_product_tags(self) -> Dict[str, str]:
        """Get product tags for search."""
        product_tags_dict = {}
        for shop in self.db.shops.values():
            for product in shop.products:
                product_tags_dict[product.product_id] = f"{product.shop_name} {product.name} {','.join(product.tags)}"
        return product_tags_dict

    def _simple_search(self, query: str, candidates: Dict[str, str], top_k: int = 100) -> List[str]:
        """Simple keyword-based search."""
        query_lower = query.lower()
        scores = []
        for item_id, text in candidates.items():
            text_lower = text.lower()
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > 0:
                scores.append((item_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scores[:top_k]]

    def _get_shop(self, shop_id: str) -> Shop:
        """Get shop by ID."""
        if shop_id not in self.db.shops:
            raise ValueError(f"Shop {shop_id} not found")
        return self.db.shops[shop_id]

    def _get_product(self, product_id: str) -> ShopProduct:
        """Get product by ID."""
        for shop in self.db.shops.values():
            for product in shop.products:
                if product.product_id == product_id:
                    return product
        raise ValueError(f"Product {product_id} not found")

    def _generate_order_id(self, user_id: str) -> str:
        """Generate unique order ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_input = f"instore_{user_id}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return f"#INSTORE#{hash_value}"

    def _generate_book_id(self, user_id: str) -> str:
        """Generate unique book ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_input = f"book_{user_id}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return f"#BOOK#{hash_value}"

    def _generate_reservation_id(self, user_id: str) -> str:
        """Generate unique reservation ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_input = f"reservation_{user_id}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return f"#RESERVATION#{hash_value}"

    def _get_current_time(self) -> str:
        """Get current time as string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _validate_time_format(self, time_str: str) -> bool:
        """Validate time format."""
        try:
            datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    @is_tool(ToolType.READ)
    def instore_shop_search_recommend(self, keywords: List[str]) -> str:
        """
        Search and recommend shops based on keywords.

        Args:
            keywords: List of keywords describing shops

        Returns:
            List of matching shops
        """
        shop_tag_dict = self._get_shop_tags()
        query = " ".join(keywords)
        selected_ids = self._simple_search(query, shop_tag_dict, top_k=100)
        selected_shops = [str(self._get_shop(shop_id)) for shop_id in selected_ids]
        return "\n".join(selected_shops) if selected_shops else "No shops found"

    @is_tool(ToolType.READ)
    def instore_product_search_recommend(self, keywords: List[str]) -> str:
        """
        Search and recommend products based on keywords.

        Args:
            keywords: List of keywords describing products

        Returns:
            List of matching products
        """
        product_tag_dict = self._get_product_tags()
        query = " ".join(keywords)
        selected_ids = self._simple_search(query, product_tag_dict, top_k=100)
        selected_products = [repr(self._get_product(product_id)) for product_id in selected_ids]
        return "\n".join(selected_products) if selected_products else "No products found"

    @is_tool(ToolType.WRITE)
    def create_instore_product_order(
        self,
        user_id: str,
        shop_id: str,
        product_id: str,
        quantity: int
    ) -> str:
        """
        Create an instore product order.

        Args:
            user_id: User ID
            shop_id: Shop ID
            product_id: Product ID
            quantity: Product quantity

        Returns:
            Order information if successful, error message otherwise
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")
        if shop_id not in self.db.shops:
            raise ValueError(f"Shop {shop_id} not found")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        product = self._get_product(product_id)
        if product.shop_id != shop_id:
            raise ValueError(f"Product {product_id} does not belong to shop {shop_id}")

        total_price = product.price * quantity

        order = Order(
            order_id=self._generate_order_id(user_id),
            order_type="instore",
            user_id=user_id,
            shop_id=shop_id,
            product_id=product_id,
            quantity=quantity,
            total_price=total_price,
            create_time=self._get_current_time(),
            update_time=self._get_current_time(),
            status="unpaid"
        )

        self.db.orders[order.order_id] = order
        return repr(order)

    @is_tool(ToolType.WRITE)
    def pay_instore_order(self, order_id: str) -> str:
        """
        Pay for an instore order.

        Args:
            order_id: Order ID

        Returns:
            Payment result
        """
        if order_id not in self.db.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.db.orders[order_id]
        if order.status == "unpaid":
            order.status = "paid"
            order.update_time = self._get_current_time()
            return "Payment successful"
        else:
            return f"Order is not in unpaid status (current status: {order.status})"

    @is_tool(ToolType.WRITE)
    def instore_cancel_order(self, order_id: str) -> str:
        """
        Cancel an instore order.

        Args:
            order_id: Order ID

        Returns:
            Cancellation result
        """
        if order_id not in self.db.orders:
            raise ValueError(f"Order {order_id} not found")

        order = self.db.orders[order_id]
        if order.status == "cancelled":
            raise ValueError(f"Order {order_id} is already cancelled")

        order.status = "cancelled"
        order.update_time = self._get_current_time()
        return f"Order {order.order_id} has been cancelled."

    @is_tool(ToolType.READ)
    def get_instore_orders(self, user_id: str) -> str:
        """
        Retrieve all instore orders for a user.

        Args:
            user_id: User ID

        Returns:
            List of orders
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")

        orders = [order for order in self.db.orders.values()
                 if order.user_id == user_id and order.order_type == "instore"]

        if not orders:
            return "No instore orders found"

        return "\n".join([repr(order) for order in orders])

    @is_tool(ToolType.WRITE)
    def instore_book(
        self,
        user_id: str,
        shop_id: str,
        time: str,
        customer_count: int
    ) -> str:
        """
        Create a seat/table booking.

        Args:
            user_id: User ID
            shop_id: Shop ID
            time: Booking time (format: yyyy-mm-dd HH:MM:SS)
            customer_count: Number of customers

        Returns:
            Booking information if successful, error message otherwise
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")
        if shop_id not in self.db.shops:
            raise ValueError(f"Shop {shop_id} not found")
        if not self._validate_time_format(time):
            raise ValueError(f"Invalid time format: {time}. Expected format: yyyy-mm-dd HH:MM:SS")
        if customer_count <= 0:
            raise ValueError("Customer count must be positive")

        shop = self._get_shop(shop_id)
        if not shop.enable_book:
            raise ValueError(f"Shop {shop_id} does not support seat booking")

        book_id = self._generate_book_id(user_id)
        status = "paid" if shop.book_price == 0 else "unpaid"

        book_info = BookInfo(
            book_id=book_id,
            shop_id=shop_id,
            book_time=time,
            customer_id=user_id,
            customer_count=customer_count,
            book_price=shop.book_price,
            status=status,
            update_time=self._get_current_time()
        )

        self.db.books[book_id] = book_info
        return repr(book_info)

    @is_tool(ToolType.WRITE)
    def pay_instore_book(self, book_id: str) -> str:
        """
        Pay for a seat booking.

        Args:
            book_id: Booking ID

        Returns:
            Payment result
        """
        if book_id not in self.db.books:
            raise ValueError(f"Booking {book_id} not found")

        book = self.db.books[book_id]
        if book.status == "unpaid":
            book.status = "paid"
            book.update_time = self._get_current_time()
            return "Payment successful"
        else:
            return f"Booking is not in unpaid status (current status: {book.status})"

    @is_tool(ToolType.WRITE)
    def instore_cancel_book(self, book_id: str) -> str:
        """
        Cancel a seat booking.

        Args:
            book_id: Booking ID

        Returns:
            Cancellation result
        """
        if book_id not in self.db.books:
            raise ValueError(f"Booking {book_id} not found")

        book = self.db.books[book_id]
        if book.status == "cancelled":
            raise ValueError(f"Booking {book_id} is already cancelled")

        book.status = "cancelled"
        book.update_time = self._get_current_time()
        return f"Booking {book.book_id} has been cancelled."

    @is_tool(ToolType.READ)
    def get_instore_books(self, user_id: str) -> str:
        """
        Retrieve all seat bookings for a user.

        Args:
            user_id: User ID

        Returns:
            List of bookings
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")

        books = [book for book in self.db.books.values() if book.customer_id == user_id]

        if not books:
            return "No seat bookings found"

        return "\n".join([repr(book) for book in books])

    @is_tool(ToolType.READ)
    def search_instore_book(self, book_id: Optional[str] = None) -> str:
        """
        Query specific or all seat bookings.

        Args:
            book_id: Optional booking ID. If not provided, returns all bookings.

        Returns:
            Booking information
        """
        if book_id:
            if book_id not in self.db.books:
                raise ValueError(f"Booking {book_id} not found")
            return repr(self.db.books[book_id])
        else:
            if not self.db.books:
                return "No seat bookings found"
            return "\n".join([repr(book) for book in self.db.books.values()])

    @is_tool(ToolType.WRITE)
    def instore_reservation(
        self,
        user_id: str,
        shop_id: str,
        time: str,
        customer_count: int
    ) -> str:
        """
        Create a service reservation.

        Args:
            user_id: User ID
            shop_id: Shop ID
            time: Reservation time (format: yyyy-mm-dd HH:MM:SS)
            customer_count: Number of customers

        Returns:
            Reservation information if successful, error message otherwise
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")
        if shop_id not in self.db.shops:
            raise ValueError(f"Shop {shop_id} not found")
        if not self._validate_time_format(time):
            raise ValueError(f"Invalid time format: {time}. Expected format: yyyy-mm-dd HH:MM:SS")
        if customer_count <= 0:
            raise ValueError("Customer count must be positive")

        shop = self._get_shop(shop_id)
        if not shop.enable_reservation:
            raise ValueError(f"Shop {shop_id} does not support service reservation")

        reservation_id = self._generate_reservation_id(user_id)

        reservation_info = ReservationInfo(
            reservation_id=reservation_id,
            shop_id=shop_id,
            reservation_time=time,
            customer_id=user_id,
            customer_count=customer_count,
            status="unpaid",
            update_time=self._get_current_time()
        )

        self.db.reservations[reservation_id] = reservation_info
        return repr(reservation_info)

    @is_tool(ToolType.WRITE)
    def instore_modify_reservation(
        self,
        reservation_id: str,
        time: Optional[str] = None,
        customer_count: Optional[int] = None
    ) -> str:
        """
        Modify an existing service reservation.

        Args:
            reservation_id: Reservation ID
            time: New reservation time (optional)
            customer_count: New customer count (optional)

        Returns:
            Modification result
        """
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")

        reservation = self.db.reservations[reservation_id]

        if reservation.status in ["consumed", "cancelled"]:
            raise ValueError(f"Cannot modify reservation with status: {reservation.status}")

        if time:
            if not self._validate_time_format(time):
                raise ValueError(f"Invalid time format: {time}. Expected format: yyyy-mm-dd HH:MM:SS")
            reservation.reservation_time = time

        if customer_count is not None:
            if customer_count <= 0:
                raise ValueError("Customer count must be positive")
            reservation.customer_count = customer_count

        reservation.update_time = self._get_current_time()
        return f"Reservation {reservation_id} has been modified."

    @is_tool(ToolType.WRITE)
    def instore_cancel_reservation(self, reservation_id: str) -> str:
        """
        Cancel a service reservation.

        Args:
            reservation_id: Reservation ID

        Returns:
            Cancellation result
        """
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")

        reservation = self.db.reservations[reservation_id]
        if reservation.status == "cancelled":
            raise ValueError(f"Reservation {reservation_id} is already cancelled")

        reservation.status = "cancelled"
        reservation.update_time = self._get_current_time()
        return f"Reservation {reservation.reservation_id} has been cancelled."

    @is_tool(ToolType.READ)
    def get_instore_reservations(self, user_id: str) -> str:
        """
        Retrieve all service reservations for a user.

        Args:
            user_id: User ID

        Returns:
            List of reservations
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")

        reservations = [res for res in self.db.reservations.values() if res.customer_id == user_id]

        if not reservations:
            return "No service reservations found"

        return "\n".join([repr(res) for res in reservations])

    @is_tool(ToolType.READ)
    def search_instore_reservation(self, reservation_id: Optional[str] = None) -> str:
        """
        Query specific or all service reservations.

        Args:
            reservation_id: Optional reservation ID. If not provided, returns all reservations.

        Returns:
            Reservation information
        """
        if reservation_id:
            if reservation_id not in self.db.reservations:
                raise ValueError(f"Reservation {reservation_id} not found")
            return repr(self.db.reservations[reservation_id])
        else:
            if not self.db.reservations:
                return "No service reservations found"
            return "\n".join([repr(res) for res in self.db.reservations.values()])
