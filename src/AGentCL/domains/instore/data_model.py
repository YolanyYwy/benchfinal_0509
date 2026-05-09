"""Data models for the instore domain."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from AGentCL.environment.db import DB


class Location(BaseModel):
    """Location information."""

    address: str = Field(default="", description="Address")
    longitude: float = Field(default=0.0, description="Longitude")
    latitude: float = Field(default=0.0, description="Latitude")

    def __repr__(self):
        return f"Location(address={self.address}, longitude={self.longitude}, latitude={self.latitude})"


class ShopProduct(BaseModel):
    """Product in a shop."""

    product_id: str = Field(default="", description="Product ID")
    name: str = Field(default="", description="Product name")
    shop_id: str = Field(default="", description="Shop ID")
    shop_name: str = Field(default="", description="Shop name")
    price: float = Field(default=0.0, description="Product price")
    quantity: int = Field(default=1, description="Product quantity")
    tags: List[str] = Field(default_factory=list, description="Product tags")

    def __repr__(self):
        return (f"ShopProduct(shop_name={self.shop_name}, "
                f"shop_id={self.shop_id}, "
                f"product_name={self.name}, "
                f"product_id={self.product_id}, "
                f"quantity={self.quantity}, "
                f"price={self.price}, "
                f"tags={self.tags})")


class Shop(BaseModel):
    """Shop information."""

    shop_id: str = Field(default="", description="Shop ID")
    shop_name: str = Field(default="", description="Shop name")
    score: float = Field(default=0.0, description="Shop rating")
    location: Location = Field(default_factory=Location, description="Shop location")
    tags: List[str] = Field(default_factory=list, description="Shop tags")
    enable_book: bool = Field(default=False, description="Whether seat booking is enabled")
    book_price: float = Field(default=0.0, description="Seat booking price")
    enable_reservation: bool = Field(default=False, description="Whether service reservation is enabled")
    products: List[ShopProduct] = Field(default_factory=list, description="List of products")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Shop(shop_name={self.shop_name}, "
                f"shop_id={self.shop_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags}, "
                f"enable_book={self.enable_book}, "
                f"book_price={self.book_price}, "
                f"enable_reservation={self.enable_reservation}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Shop(shop_name={self.shop_name}, "
                f"shop_id={self.shop_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags}, "
                f"enable_book={self.enable_book}, "
                f"book_price={self.book_price}, "
                f"enable_reservation={self.enable_reservation})")


class Order(BaseModel):
    """Order information."""

    order_id: str = Field(description="Order ID")
    order_type: str = Field(default="", description="Order type")
    user_id: str = Field(default="", description="User ID")
    shop_id: str = Field(default="", description="Shop ID")
    product_id: str = Field(default="", description="Product ID")
    quantity: int = Field(default=1, description="Product quantity")
    total_price: float = Field(default=0.0, description="Total price")
    create_time: str = Field(default="", description="Order creation time")
    update_time: str = Field(default="", description="Order update time")
    status: str = Field(default="pending", description="Order status")

    def __repr__(self):
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"shop_id={self.shop_id}, "
                f"product_id={self.product_id}, "
                f"quantity={self.quantity}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time}, "
                f"status={self.status})")


class BookInfo(BaseModel):
    """Seat/table booking information."""

    book_id: str = Field(description="Booking ID")
    shop_id: str = Field(default="", description="Shop ID")
    book_time: str = Field(default="", description="Booking time")
    customer_id: str = Field(default="", description="Customer ID")
    customer_count: int = Field(default=1, description="Number of customers")
    book_price: float = Field(default=0.0, description="Booking price")
    status: str = Field(default="pending", description="Booking status")
    update_time: str = Field(default="", description="Last update time")

    def __repr__(self):
        return (f"BookInfo(book_id={self.book_id}, "
                f"shop_id={self.shop_id}, "
                f"book_time={self.book_time}, "
                f"customer_id={self.customer_id}, "
                f"customer_count={self.customer_count}, "
                f"book_price={self.book_price}, "
                f"status={self.status}, "
                f"update_time={self.update_time})")


class ReservationInfo(BaseModel):
    """Service reservation information."""

    reservation_id: str = Field(description="Reservation ID")
    shop_id: str = Field(default="", description="Shop ID")
    reservation_time: str = Field(default="", description="Reservation time")
    customer_id: str = Field(default="", description="Customer ID")
    customer_count: int = Field(default=1, description="Number of customers")
    status: str = Field(default="pending", description="Reservation status")
    update_time: str = Field(default="", description="Last update time")

    def __repr__(self):
        return (f"ReservationInfo(reservation_id={self.reservation_id}, "
                f"shop_id={self.shop_id}, "
                f"reservation_time={self.reservation_time}, "
                f"customer_id={self.customer_id}, "
                f"customer_count={self.customer_count}, "
                f"status={self.status}, "
                f"update_time={self.update_time})")


class InstoreDB(DB):
    """Database for the instore domain."""

    shops: Dict[str, Shop] = Field(
        default_factory=dict,
        description="Dictionary of shops indexed by shop ID"
    )
    orders: Dict[str, Order] = Field(
        default_factory=dict,
        description="Dictionary of orders indexed by order ID"
    )
    books: Dict[str, BookInfo] = Field(
        default_factory=dict,
        description="Dictionary of seat bookings indexed by book ID"
    )
    reservations: Dict[str, ReservationInfo] = Field(
        default_factory=dict,
        description="Dictionary of service reservations indexed by reservation ID"
    )
    time: Optional[str] = Field(
        default=None,
        description="Current time"
    )
    weather: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Weather information"
    )
    location: Optional[List[Location]] = Field(
        default=None,
        description="User location information"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Current user ID"
    )
    user_historical_behaviors: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User historical behaviors"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics of the database."""
        return {
            "num_shops": len(self.shops),
            "num_orders": len(self.orders),
            "num_books": len(self.books),
            "num_reservations": len(self.reservations),
        }
