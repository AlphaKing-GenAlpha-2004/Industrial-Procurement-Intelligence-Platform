import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

rows = 50000
start_date = datetime(2023, 1, 1)

suppliers = [
    "Supplier_A",
    "Supplier_B",
    "Supplier_C",
    "Supplier_D",
    "Supplier_E"
]

warehouses = [
    "Warehouse_North",
    "Warehouse_South",
    "Warehouse_East",
    "Warehouse_West"
]

item_categories = [
    "Electronics",
    "Mechanical",
    "Packaging",
    "Chemicals",
    "Raw_Materials"
]

procurement_statuses = [
    "Approved",
    "Pending",
    "Delayed"
]

transport_modes = [
    "Truck",
    "Rail",
    "Ship",
    "Air"
]

data = []

for i in range(rows):

    procurement_quantity = np.random.randint(
        50,
        5000
    )

    inventory_level = np.random.randint(
        100,
        10000
    )

    reorder_point = np.random.randint(
        100,
        3000
    )

    supplier_lead_time = np.random.randint(
        1,
        40
    )

    transportation_cost = round(
        np.random.uniform(500, 15000),
        2
    )

    fuel_cost = round(
        np.random.uniform(1000, 8000),
        2
    )

    holding_cost = round(
        inventory_level
        *
        np.random.uniform(0.3, 2.0),
        2
    )

    damaged_inventory_percent = round(
        np.random.uniform(0, 10),
        2
    )

    stockout_frequency = np.random.randint(
        0,
        20
    )

    supplier_reliability = round(
        np.random.uniform(50, 100),
        2
    )

    market_demand_index = round(
        np.random.uniform(50, 150),
        2
    )

    warehouse_utilization = round(
        np.random.uniform(50, 100),
        2
    )

    procurement_cost = round(

        procurement_quantity
        *
        np.random.uniform(5, 25)

        +

        transportation_cost * 0.6

        +

        fuel_cost * 0.5

        +

        holding_cost * 0.3,

        2
    )

    historical_demand = round(

        procurement_quantity * 1.5

        +

        market_demand_index * 18

        -

        supplier_lead_time * 25

        -

        damaged_inventory_percent * 50

        +

        np.random.normal(0, 200),

        2
    )

    # =====================================================
    # RISK LOGIC
    # =====================================================

    risk_score = (

        supplier_lead_time * 1.8

        +

        damaged_inventory_percent * 2

        +

        stockout_frequency * 1.5

        -

        supplier_reliability * 0.5
    )

    if risk_score > 45:

        safety_risk = "High"

    elif risk_score > 25:

        safety_risk = "Medium"

    else:

        safety_risk = "Low"

    # =====================================================
    # PROCUREMENT STATUS
    # =====================================================

    if supplier_lead_time > 25:

        procurement_status = "Delayed"

    elif supplier_lead_time > 12:

        procurement_status = "Pending"

    else:

        procurement_status = "Approved"

    data.append({

        "date": (
            start_date + timedelta(days=i % 730)
        ).strftime("%Y-%m-%d"),

        "supplier_name":
        random.choice(suppliers),

        "warehouse_name":
        random.choice(warehouses),

        "item_category":
        random.choice(item_categories),

        "transport_mode":
        random.choice(transport_modes),

        "procurement_status":
        procurement_status,

        "procurement_quantity":
        procurement_quantity,

        "inventory_level":
        inventory_level,

        "reorder_point":
        reorder_point,

        "supplier_lead_time":
        supplier_lead_time,

        "transportation_cost":
        transportation_cost,

        "fuel_cost":
        fuel_cost,

        "holding_cost":
        holding_cost,

        "damaged_inventory_percent":
        damaged_inventory_percent,

        "stockout_frequency":
        stockout_frequency,

        "supplier_reliability":
        supplier_reliability,

        "market_demand_index":
        market_demand_index,

        "warehouse_utilization":
        warehouse_utilization,

        "procurement_cost":
        procurement_cost,

        "historical_demand":
        historical_demand,

        "safety_risk":
        safety_risk
    })

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(data)

# =========================================================
# ENSURE NO MISSING VALUES
# =========================================================

df = df.fillna(0)

# =========================================================
# SAVE DATASET
# =========================================================

df.to_csv(
    "procurement_dataset_50k.csv",
    index=False
)

print(df.head())

print("\nDataset Shape:", df.shape)

print(
    "\nMissing Values:",
    df.isnull().sum().sum()
)

print(
    "\nProcurement Dataset Generated Successfully."
)