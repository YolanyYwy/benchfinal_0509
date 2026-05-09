"""Tools for the delivery domain."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import math

from AGentCL.domains.delivery.data_model import DeliveryDB, Order, Store, StoreProduct, Location
from AGentCL.environment.toolkit import ToolKitBase, ToolType, is_tool


class DeliveryTools(ToolKitBase):
    """Tools for the delivery domain."""

    db: DeliveryDB

    def __init__(self, db: DeliveryDB):
        super().__init__(db)

    def _check_user(self, user_id: str) -> bool:
        """Check if user_id is valid (placeholder - always returns True)."""
        return True

    def _get_store_tags(self) -> Dict[str, str]:
        """Get store tags for search."""
        tag_dict = {}
        for store in self.db.stores.values():
            tag_dict[store.store_id] = store.name + ',' + ','.join(store.tags)
        return tag_dict

    def _get_store_product_tags(self) -> Dict[str, str]:
        """Get product tags for search."""
        product_tags_dict = {}
        for store in self.db.stores.values():
            for product in store.products:
                product_tags_dict[product.product_id] = f"{product.store_name} {product.name} {','.join(product.tags)}"
        return product_tags_dict

    def _get_delivery_order(self, order_id: Optional[str] = None):
        """Get delivery order(s)."""
        if order_id is None:
            return [order for order in self.db.orders.values() if order.order_type == "delivery"]
        if order_id not in self.db.orders:
            raise ValueError(f"Order {order_id} not found")
        order = self.db.orders[order_id]
        if order.order_type != "delivery":
            raise ValueError(f"Order {order_id} is not a delivery order")
        return order

    def _add_delivery_order(self, order: Order) -> str:
        """Add order to database."""
        if order.order_id in self.db.orders:
            return "Order already exists"
        self.db.orders[order.order_id] = order
        return "done"

    def _modify_delivery_order(self, order: Order) -> str:
        """Modify order in database."""
        if order.order_id not in self.db.orders:
            return "Order not found"
        self.db.orders[order.order_id] = order
        return "done"

    def _get_store(self, store_id: str) -> Store:
        """Get store by ID."""
        if store_id not in self.db.stores:
            raise ValueError(f"Store {store_id} not found")
        return self.db.stores[store_id]

    def _get_store_product(self, product_id: str) -> StoreProduct:
        """Get product by ID."""
        for store in self.db.stores.values():
            for product in store.products:
                if product.product_id == product_id:
                    return product
        raise ValueError(f"Product {product_id} not found")

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

    def _calculate_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate distance between two coordinates in meters."""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _generate_order_id(self, user_id: str) -> str:
        """Generate unique order ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_input = f"delivery_{user_id}_{timestamp}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return f"#DELIVERY#{hash_value}"

    def _get_current_time(self) -> str:
        """Get current time as string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    def _format_time(self, dt: datetime) -> str:
        """Format datetime to string."""
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @is_tool(ToolType.GENERIC)
    def delivery_distance_to_time(self, distance: float) -> float:
        """
        Calculate delivery time (minutes) based on distance (meters).

        Args:
            distance: Distance in meters

        Returns:
            Delivery time in minutes
        """
        if not isinstance(distance, (float, int)):
            raise ValueError(f"distance must be float or int, got {type(distance)}")
        return round(25.00 + int(distance) * 0.006510, 0)

    @is_tool(ToolType.READ)
    def get_delivery_store_info(self, store_id: str) -> str:
        """
        Get store information including store id, rating, address, longitude, latitude, tags, and product list.

        Args:
            store_id: Store ID

        Returns:
            Detailed store information
        """
        store = self._get_store(store_id)
        return repr(store)

    @is_tool(ToolType.READ)
    def get_delivery_product_info(self, product_id: str) -> str:
        """
        Get product information including product name, product id, store name, store id, price, and tags.

        Args:
            product_id: Product ID

        Returns:
            Detailed product information
        """
        product = self._get_store_product(product_id)
        return repr(product)

    @is_tool(ToolType.READ)
    def delivery_store_search_recommand(self, keywords: List[str]) -> str:
        """
        Search or recommend stores based on keywords.

        Args:
            keywords: List of keywords describing stores

        Returns:
            List of matching stores
        """
        store_tag_dict = self._get_store_tags()
        query = " ".join(keywords)
        selected_ids = self._simple_search(query, store_tag_dict, top_k=100)
        selected_stores = [str(self._get_store(store_id)) for store_id in selected_ids]
        return "\n".join(selected_stores) if selected_stores else "No stores found"

    @is_tool(ToolType.READ)
    def delivery_product_search_recommand(self, keywords: List[str]) -> str:
        """
        Search or recommend products based on keywords.

        Args:
            keywords: List of keywords describing products

        Returns:
            List of matching products
        """
        product_tag_dict = self._get_store_product_tags()
        query = " ".join(keywords)
        selected_ids = self._simple_search(query, product_tag_dict, top_k=100)
        selected_products = [repr(self._get_store_product(product_id)) for product_id in selected_ids]
        return "\n".join(selected_products) if selected_products else "No products found"

    @is_tool(ToolType.WRITE)
    def create_delivery_order(
        self,
        user_id: str,
        store_id: str,
        product_ids: List[str],
        product_cnts: List[int],
        address: str,
        dispatch_time: str,
        attributes: Optional[List[str]] = None,
        note: str = "",
    ) -> str:
        """
        Create a delivery order. Only supports single store orders, but can order multiple products from that store.

        Args:
            user_id: User ID
            store_id: Store ID
            product_ids: List of product IDs
            product_cnts: List of quantities corresponding to product_ids
            address: Delivery address
            dispatch_time: Dispatch time in format yyyy-mm-dd HH:MM:SS
            attributes: Optional list of product attributes
            note: Order note (e.g., dietary restrictions)

        Returns:
            Order information if successful, error message otherwise
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")
        if store_id not in self.db.stores:
            raise ValueError(f"Store {store_id} not found")
        if not address:
            raise ValueError("Address cannot be empty")
        if len(product_ids) != len(product_cnts):
            raise ValueError("product_ids and product_cnts must have same length")
        if any(cnt <= 0 for cnt in product_cnts):
            raise ValueError("All product counts must be positive")

        # Validate dispatch time
        try:
            dispatch_dt = self._parse_time(dispatch_time)
            current_dt = self._parse_time(self._get_current_time())
            if dispatch_dt < current_dt:
                raise ValueError(f"dispatch_time {dispatch_time} must be in the future")
        except ValueError as e:
            raise ValueError(f"Invalid dispatch_time format: {e}")

        # Get products and store
        products = [self._get_store_product(pid) for pid in product_ids]
        store = self._get_store(store_id)

        # Calculate delivery time (simplified - assumes address has coordinates)
        # For now, use a default distance calculation
        distance = 2000  # Default 2km
        shipping_time = self.delivery_distance_to_time(distance)
        delivery_dt = dispatch_dt + timedelta(minutes=shipping_time)
        delivery_time = self._format_time(delivery_dt)

        # Calculate total price
        total_price = sum(product.price * cnt for product, cnt in zip(products, product_cnts))

        # Prepare attributes
        attribute_list = [""] * len(products)
        if attributes:
            for i, attr in enumerate(attributes[:len(products)]):
                if attr:
                    attribute_list[i] = attr

        # Create ordered products
        ordered_products = []
        for product, cnt, attr in zip(products, product_cnts, attribute_list):
            ordered_product = StoreProduct(
                product_id=product.product_id,
                name=product.name,
                store_id=product.store_id,
                store_name=product.store_name,
                price=product.price,
                quantity=cnt,
                attributes=attr,
                tags=product.tags
            )
            ordered_products.append(ordered_product)

        # Create order
        order = Order(
            order_id=self._generate_order_id(user_id),
            order_type="delivery",
            user_id=user_id,
            store_id=store_id,
            location=Location(address=address, longitude=0.0, latitude=0.0),
            dispatch_time=dispatch_time,
            shipping_time=shipping_time,
            delivery_time=delivery_time,
            total_price=total_price,
            create_time=self._get_current_time(),
            update_time=self._get_current_time(),
            note=note,
            products=ordered_products,
            status="unpaid"
        )

        response = self._add_delivery_order(order)
        return repr(order) if response == "done" else response

    @is_tool(ToolType.WRITE)
    def pay_delivery_order(self, order_id: str) -> str:
        """
        Pay for a delivery order.

        Args:
            order_id: Order ID

        Returns:
            Payment result
        """
        order = self._get_delivery_order(order_id)
        if order.status == "unpaid":
            order.status = "paid"
            order.update_time = self._get_current_time()
            self._modify_delivery_order(order)
            return "Payment successful"
        else:
            return "Order is not in unpaid status"

    @is_tool(ToolType.READ)
    def get_delivery_order_status(self, order_id: str) -> str:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order status
        """
        order = self._get_delivery_order(order_id)
        return order.status

    @is_tool(ToolType.WRITE)
    def cancel_delivery_order(self, order_id: str) -> str:
        """
        Cancel a delivery order. Cannot cancel orders that are already cancelled.

        Args:
            order_id: Order ID

        Returns:
            Cancellation result
        """
        order = self._get_delivery_order(order_id)
        if order.status == "cancelled":
            raise ValueError(f"Order {order_id} is already cancelled")
        order.status = "cancelled"
        order.update_time = self._get_current_time()
        resp = self._modify_delivery_order(order)
        if resp == "done":
            return f"Order {order.order_id} has been cancelled."
        else:
            return resp

    @is_tool(ToolType.WRITE)
    def modify_delivery_order(self, order_id: str, note: str) -> str:
        """
        Modify order note information.

        Args:
            order_id: Order ID
            note: New order note

        Returns:
            Modification result
        """
        order = self._get_delivery_order(order_id)
        order.note = note
        order.update_time = self._get_current_time()
        resp = self._modify_delivery_order(order)
        if resp == "done":
            return f"Order {order.order_id} note has been modified."
        else:
            return resp

    @is_tool(ToolType.READ)
    def search_delivery_orders(self, user_id: str, status: str = "unpaid") -> str:
        """
        Search delivery orders by user ID and status.

        Args:
            user_id: User ID
            status: Order status (default: unpaid)

        Returns:
            List of matching orders
        """
        if not user_id:
            raise ValueError("User ID cannot be empty")
        if not self._check_user(user_id):
            raise ValueError("User ID does not match")

        delivery_orders = []
        for order in self._get_delivery_order():
            if order.order_type == "delivery" and order.status == status and order.user_id == user_id:
                delivery_orders.append(order)

        if not delivery_orders:
            return "No delivery orders available"

        return "\n".join([str(order) for order in delivery_orders])

    @is_tool(ToolType.READ)
    def get_delivery_order_detail(self, order_id: str) -> str:
        """
        Get detailed order information by order ID.

        Args:
            order_id: Order ID

        Returns:
            Detailed order information
        """
        if not order_id:
            raise ValueError("Order ID cannot be empty")
        order = self._get_delivery_order(order_id)
        return repr(order)
