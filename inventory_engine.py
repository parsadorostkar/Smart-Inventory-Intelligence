import math


def _validate_non_negative(values, message):
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError(message)


def calculate_reorder_point(
    forecast_demand_during_lead_time: float,
    safety_stock: float
) -> float:
    _validate_non_negative(
        [forecast_demand_during_lead_time, safety_stock],
        "Forecast demand and safety stock must be finite and non-negative."
    )
    return forecast_demand_during_lead_time + safety_stock


def calculate_inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float
) -> float:
    _validate_non_negative(
        [on_hand, on_order, backorders],
        "Inventory values must be finite and non-negative."
    )
    return on_hand + on_order - backorders


def recommend_order(
    inventory_position: float,
    reorder_point: float,
    order_up_to_level: float
) -> dict:
    if not all(math.isfinite(v) for v in [
        inventory_position, reorder_point, order_up_to_level
    ]):
        raise ValueError("Inventory values must be finite.")

    if reorder_point < 0 or order_up_to_level < reorder_point:
        raise ValueError(
            "Reorder point must be non-negative, and order-up-to level "
            "must be greater than or equal to reorder point."
        )

    if inventory_position > reorder_point:
        return {"should_order": False, "order_quantity": 0}

    order_quantity = max(0, order_up_to_level - inventory_position)

    return {
        "should_order": order_quantity > 0,
        "order_quantity": order_quantity
    }


def calculate_order_up_to_level(
    forecast_demand_during_protection_period: float,
    safety_stock: float
) -> float:
    _validate_non_negative(
        [forecast_demand_during_protection_period, safety_stock],
        "Forecast demand and safety stock must be finite and non-negative."
    )
    return forecast_demand_during_protection_period + safety_stock