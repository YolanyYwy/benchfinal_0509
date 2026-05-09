"""Data models for the ota domain."""

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


class HotelProduct(BaseModel):
    """Hotel room product."""

    product_id: str = Field(default="", description="Product ID")
    room_type: str = Field(default="", description="Room type")
    date: str = Field(default="", description="Check-in date")
    price: float = Field(default=0.0, description="Room price")
    quantity: int = Field(default=1, description="Available quantity")

    def __repr__(self):
        return (f"HotelProduct(product_id={self.product_id}, "
                f"room_type={self.room_type}, "
                f"date={self.date}, "
                f"price={self.price}, "
                f"quantity={self.quantity})")


class Hotel(BaseModel):
    """Hotel information."""

    hotel_id: str = Field(default="", description="Hotel ID")
    hotel_name: str = Field(default="", description="Hotel name")
    score: float = Field(default=0.0, description="Hotel rating")
    star_rating: int = Field(default=0, description="Star rating (0-5)")
    location: Location = Field(default_factory=Location, description="Hotel location")
    tags: List[str] = Field(default_factory=list, description="Hotel tags")
    products: List[HotelProduct] = Field(default_factory=list, description="Available rooms")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Hotel(hotel_name={self.hotel_name}, "
                f"hotel_id={self.hotel_id}, "
                f"score={self.score}, "
                f"star_rating={self.star_rating}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Hotel(hotel_name={self.hotel_name}, "
                f"hotel_id={self.hotel_id}, "
                f"score={self.score}, "
                f"star_rating={self.star_rating}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags})")


class AttractionProduct(BaseModel):
    """Attraction ticket product."""

    product_id: str = Field(default="", description="Product ID")
    ticket_type: str = Field(default="", description="Ticket type")
    date: str = Field(default="", description="Visit date")
    price: float = Field(default=0.0, description="Ticket price")
    quantity: int = Field(default=1, description="Available quantity")

    def __repr__(self):
        return (f"AttractionProduct(product_id={self.product_id}, "
                f"ticket_type={self.ticket_type}, "
                f"date={self.date}, "
                f"price={self.price}, "
                f"quantity={self.quantity})")


class Attraction(BaseModel):
    """Attraction information."""

    attraction_id: str = Field(default="", description="Attraction ID")
    attraction_name: str = Field(default="", description="Attraction name")
    location: Location = Field(default_factory=Location, description="Attraction location")
    description: str = Field(default="", description="Attraction description")
    score: float = Field(default=0.0, description="Attraction rating")
    opening_hours: str = Field(default="", description="Opening hours")
    ticket_price: float = Field(default=0.0, description="Base ticket price")
    products: List[AttractionProduct] = Field(default_factory=list, description="Available tickets")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Attraction(attraction_name={self.attraction_name}, "
                f"attraction_id={self.attraction_id}, "
                f"location={repr(self.location)}, "
                f"description={self.description}, "
                f"score={self.score}, "
                f"opening_hours={self.opening_hours}, "
                f"ticket_price={self.ticket_price}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Attraction(attraction_name={self.attraction_name}, "
                f"attraction_id={self.attraction_id}, "
                f"location={repr(self.location)}, "
                f"score={self.score}, "
                f"ticket_price={self.ticket_price})")


class FlightProduct(BaseModel):
    """Flight seat product."""

    product_id: str = Field(default="", description="Product ID")
    seat_type: str = Field(default="", description="Seat type")
    date: str = Field(default="", description="Flight date")
    price: float = Field(default=0.0, description="Seat price")
    quantity: int = Field(default=1, description="Available quantity")

    def __repr__(self):
        return (f"FlightProduct(product_id={self.product_id}, "
                f"seat_type={self.seat_type}, "
                f"date={self.date}, "
                f"price={self.price}, "
                f"quantity={self.quantity})")


class Flight(BaseModel):
    """Flight information."""

    flight_id: str = Field(default="", description="Flight ID")
    flight_number: str = Field(default="", description="Flight number")
    departure_city: str = Field(default="", description="Departure city")
    arrival_city: str = Field(default="", description="Arrival city")
    departure_airport_location: Location = Field(default_factory=Location, description="Departure airport location")
    arrival_airport_location: Location = Field(default_factory=Location, description="Arrival airport location")
    departure_time: str = Field(default="", description="Departure time")
    arrival_time: str = Field(default="", description="Arrival time")
    tags: List[str] = Field(default_factory=list, description="Flight tags")
    products: List[FlightProduct] = Field(default_factory=list, description="Available seats")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Flight(flight_number={self.flight_number}, "
                f"flight_id={self.flight_id}, "
                f"departure_city={self.departure_city}, "
                f"arrival_city={self.arrival_city}, "
                f"departure_airport={repr(self.departure_airport_location)}, "
                f"arrival_airport={repr(self.arrival_airport_location)}, "
                f"departure_time={self.departure_time}, "
                f"arrival_time={self.arrival_time}, "
                f"tags={self.tags}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Flight(flight_number={self.flight_number}, "
                f"flight_id={self.flight_id}, "
                f"departure_city={self.departure_city}, "
                f"arrival_city={self.arrival_city}, "
                f"departure_time={self.departure_time}, "
                f"arrival_time={self.arrival_time})")


class TrainProduct(BaseModel):
    """Train seat product."""

    product_id: str = Field(default="", description="Product ID")
    seat_type: str = Field(default="", description="Seat type")
    date: str = Field(default="", description="Train date")
    price: float = Field(default=0.0, description="Seat price")
    quantity: int = Field(default=1, description="Available quantity")

    def __repr__(self):
        return (f"TrainProduct(product_id={self.product_id}, "
                f"seat_type={self.seat_type}, "
                f"date={self.date}, "
                f"price={self.price}, "
                f"quantity={self.quantity})")


class Train(BaseModel):
    """Train information."""

    train_id: str = Field(default="", description="Train ID")
    train_number: str = Field(default="", description="Train number")
    departure_city: str = Field(default="", description="Departure city")
    arrival_city: str = Field(default="", description="Arrival city")
    departure_station_location: Location = Field(default_factory=Location, description="Departure station location")
    arrival_station_location: Location = Field(default_factory=Location, description="Arrival station location")
    departure_time: str = Field(default="", description="Departure time")
    arrival_time: str = Field(default="", description="Arrival time")
    tags: List[str] = Field(default_factory=list, description="Train tags")
    products: List[TrainProduct] = Field(default_factory=list, description="Available seats")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Train(train_number={self.train_number}, "
                f"train_id={self.train_id}, "
                f"departure_city={self.departure_city}, "
                f"arrival_city={self.arrival_city}, "
                f"departure_station={repr(self.departure_station_location)}, "
                f"arrival_station={repr(self.arrival_station_location)}, "
                f"departure_time={self.departure_time}, "
                f"arrival_time={self.arrival_time}, "
                f"tags={self.tags}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Train(train_number={self.train_number}, "
                f"train_id={self.train_id}, "
                f"departure_city={self.departure_city}, "
                f"arrival_city={self.arrival_city}, "
                f"departure_time={self.departure_time}, "
                f"arrival_time={self.arrival_time})")


class Order(BaseModel):
    """Generic order information."""

    order_id: str = Field(description="Order ID")
    order_type: str = Field(default="", description="Order type (hotel/attraction/flight/train)")
    user_id: str = Field(default="", description="User ID")
    entity_id: str = Field(default="", description="Entity ID (hotel/attraction/flight/train)")
    product_id: str = Field(default="", description="Product ID")
    quantity: int = Field(default=1, description="Quantity")
    total_price: float = Field(default=0.0, description="Total price")
    create_time: str = Field(default="", description="Order creation time")
    update_time: str = Field(default="", description="Order update time")
    status: str = Field(default="pending", description="Order status")

    def __repr__(self):
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"entity_id={self.entity_id}, "
                f"product_id={self.product_id}, "
                f"quantity={self.quantity}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time}, "
                f"status={self.status})")


class OTADB(DB):
    """Database for the ota domain."""

    hotels: Dict[str, Hotel] = Field(
        default_factory=dict,
        description="Dictionary of hotels indexed by hotel ID"
    )
    attractions: Dict[str, Attraction] = Field(
        default_factory=dict,
        description="Dictionary of attractions indexed by attraction ID"
    )
    flights: Dict[str, Flight] = Field(
        default_factory=dict,
        description="Dictionary of flights indexed by flight ID"
    )
    trains: Dict[str, Train] = Field(
        default_factory=dict,
        description="Dictionary of trains indexed by train ID"
    )
    orders: Dict[str, Order] = Field(
        default_factory=dict,
        description="Dictionary of orders indexed by order ID"
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
            "num_hotels": len(self.hotels),
            "num_attractions": len(self.attractions),
            "num_flights": len(self.flights),
            "num_trains": len(self.trains),
            "num_orders": len(self.orders),
        }
