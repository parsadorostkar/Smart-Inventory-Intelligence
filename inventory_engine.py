
import math


def calculate_reorder_point(
    forecast_demand_during_lead_time: float,
    safety_stock: float
) -> float:
    values = [
        forecast_demand_during_lead_time,
        safety_stock
    ]

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Demand forecast and safety stock must be finite and non-negative.")

    return forecast_demand_during_lead_time + safety_stock


rop = calculate_reorder_point(
    forecast_demand_during_lead_time=120,
    safety_stock=30
)

print(rop)



def calculate_inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float
) -> float:
    values = [on_hand, on_order, backorders]

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Inventory values must be finite and non-negative.")

    return on_hand + on_order - backorders

position = calculate_inventory_position(
    on_hand=80,
    on_order=40,
    backorders=10
)

print(position)


def recommend_order(
    inventory_position: float,
    reorder_point: float,
    order_up_to_level: float
) -> dict:
    values = [
        inventory_position,
        reorder_point,
        order_up_to_level
    ]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Inventory values must be finite.")

    if reorder_point < 0 or order_up_to_level < reorder_point:
        raise ValueError(
            "Reorder point must be non-negative, and order-up-to level "
            "must be greater than or equal to reorder point."
        )

    if inventory_position > reorder_point:
        return {
            "should_order": False,
            "order_quantity": 0
        }

    order_quantity = max(0, order_up_to_level - inventory_position)

    return {
        "should_order": order_quantity > 0,
        "order_quantity": order_quantity
    }

result = recommend_order(
    inventory_position=110,
    reorder_point=150,
    order_up_to_level=200
)

print(result)


def calculate_order_up_to_level(
    forecast_demand_during_protection_period: float,
    safety_stock: float
) -> float:
    values = [
        forecast_demand_during_protection_period,
        safety_stock
    ]

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(
            "Forecast demand and safety stock must be finite and non-negative."
        )

    return forecast_demand_during_protection_period + safety_stock

target = calculate_order_up_to_level(
    forecast_demand_during_protection_period=160,
    safety_stock=30
)

print(target)


rop = calculate_reorder_point(120, 30)
target = calculate_order_up_to_level(160, 30)

position = calculate_inventory_position(
    on_hand=80,
    on_order=40,
    backorders=10
)

result = recommend_order(
    inventory_position=position,
    reorder_point=rop,
    order_up_to_level=target
)

print("Reorder Point:", rop)
print("Inventory Position:", position)
print("Order-up-to Level:", target)
print("Decision:", result)
