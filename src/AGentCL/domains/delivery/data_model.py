"""Data models for the delivery domain."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from AGentCL.environment.db import DB


class Location(BaseModel):
    """Location information."""

    address: str = Field(default="", description="Address")
    longitude: float = Field(default=0.0, description="Longitude")
    latitude: float = Field(default=0.0, description="Latitude")

    def __repr__(self):
        return f"Location(address={self.address}, longitude={self.longitude}, latitude={self.latitude})"


class StoreProduct(BaseModel):
    """Product in a store."""

    product_id: str = Field(default="", description="Product ID")
    name: str = Field(default="", description="Product name")
    store_id: str = Field(default="", description="Store ID")
    store_name: str = Field(default="", description="Store name")
    price: float = Field(default=0.0, description="Product price")
    quantity: int = Field(default=1, description="Product quantity")
    attributes: str = Field(default="", description="Product attributes")
    tags: List[str] = Field(default_factory=list, description="Product tags")

    @field_validator('attributes', mode='before')
    @classmethod
    def convert_attributes_to_string(cls, v):
        """Convert attributes from list to string if needed."""
        if isinstance(v, list):
            return "" if not v else ", ".join(str(item) for item in v)
        return v if v is not None else ""

    def __repr__(self):
        return (f"StoreProduct(store_name={self.store_name}, "
                f"store_id={self.store_id}, "
                f"product_name={self.name}, "
                f"product_id={self.product_id}, "
                f"attributes={self.attributes}, "
                f"quantity={self.quantity}, "
                f"price={self.price}, "
                f"tags={self.tags})")


class Store(BaseModel):
    """Store information."""

    store_id: str = Field(description="Store ID")
    name: str = Field(description="Store name")
    score: float = Field(description="Store rating")
    location: Location = Field(description="Store location")
    tags: List[str] = Field(description="Store tags")
    products: List[StoreProduct] = Field(description="List of products")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Store(name={self.name}, "
                f"store_id={self.store_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Store(name={self.name}, "
                f"store_id={self.store_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags})")


class Order(BaseModel):
    """Order information."""

    order_id: str = Field(description="Order ID")
    order_type: str = Field(default="", description="Order type")
    user_id: str = Field(default="", description="User ID")
    store_id: str = Field(default="", description="Store ID")
    location: Location = Field(default_factory=Location, description="Delivery location")
    dispatch_time: str = Field(default="", description="Dispatch time")
    shipping_time: float = Field(default=0.0, description="Shipping time in minutes")
    delivery_time: str = Field(default="", description="Delivery time")
    total_price: float = Field(default=0.0, description="Total price")
    create_time: str = Field(default="", description="Order creation time")
    update_time: str = Field(default="", description="Order update time")
    note: str = Field(default="", description="Order note")
    products: List[StoreProduct] = Field(default_factory=list, description="Ordered products")
    status: str = Field(default="pending", description="Order status")

    @field_validator('shipping_time', mode='before')
    @classmethod
    def convert_shipping_time(cls, v):
        """Convert shipping_time to float if needed."""
        if v is None:
            return 0.0
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 0.0
        return float(v)

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"store_id={self.store_id}, "
                f"location={repr(self.location)}, "
                f"dispatch_time={self.dispatch_time}, "
                f"shipping_time={self.shipping_time}, "
                f"delivery_time={self.delivery_time}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time}, "
                f"note={self.note}, "
                f"status={self.status}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"store_id={self.store_id}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"status={self.status})")


class DeliveryDB(DB):
    """Database for the delivery domain."""

    stores: Dict[str, Store] = Field(
        default_factory=dict,
        description="Dictionary of stores indexed by store ID"
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
            "num_stores": len(self.stores),
            "num_orders": len(self.orders),
        }
