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

regions = [
    "Asia",
    "Europe",
    "North_America",
    "Middle_East",
    "Africa"
]

transport_modes = [
    "Truck",
    "Rail",
    "Ship",
    "Air"
]

materials = [
    "Steel",
    "Electronics",
    "Chemicals",
    "Packaging",
    "Machinery"
]

data = []

for i in range(rows):

    supplier_reliability = round(
        np.random.uniform(50, 100),
        2
    )

    delivery_delay_days = np.random.randint(0, 30)

    transportation_cost = round(
        np.random.uniform(500, 10000),
        2
    )

    defect_rate = round(
        np.random.uniform(0, 12),
        2
    )

    geopolitical_risk = round(
        np.random.uniform(0, 10),
        2
    )

    fuel_price_index = round(
        np.random.uniform(70, 150),
        2
    )

    inventory_buffer = np.random.randint(
        50,
        5000
    )

    order_volume = np.random.randint(
        100,
        5000
    )

    lead_time_days = np.random.randint(
        1,
        45
    )

    market_demand_index = round(
        np.random.uniform(50, 150),
        2
    )

    procurement_cost = round(

        order_volume * np.random.uniform(2, 8)

        +

        transportation_cost * 0.5

        +

        fuel_price_index * 20,

        2
    )

    historical_demand = round(

        order_volume * 1.4

        +

        market_demand_index * 15

        -

        delivery_delay_days * 20

        -

        defect_rate * 40

        +

        np.random.normal(0, 150),

        2
    )

    # =====================================================
    # SUPPLY RISK LOGIC
    # =====================================================

    risk_score = (

        delivery_delay_days * 1.8

        +

        defect_rate * 2.2

        +

        geopolitical_risk * 2

        +

        lead_time_days * 1.2

        -

        supplier_reliability * 0.5
    )

    if risk_score > 45:

        safety_risk = "High"

    elif risk_score > 25:

        safety_risk = "Medium"

    else:

        safety_risk = "Low"

    data.append({

        "date": (
            start_date + timedelta(days=i % 730)
        ).strftime("%Y-%m-%d"),

        "supplier_name": random.choice(
            suppliers
        ),

        "supplier_region": random.choice(
            regions
        ),

        "material_type": random.choice(
            materials
        ),

        "transport_mode": random.choice(
            transport_modes
        ),

        "supplier_reliability":
        supplier_reliability,

        "delivery_delay_days":
        delivery_delay_days,

        "transportation_cost":
        transportation_cost,

        "defect_rate":
        defect_rate,

        "geopolitical_risk":
        geopolitical_risk,

        "fuel_price_index":
        fuel_price_index,

        "inventory_buffer":
        inventory_buffer,

        "order_volume":
        order_volume,

        "lead_time_days":
        lead_time_days,

        "market_demand_index":
        market_demand_index,

        "procurement_cost":
        procurement_cost,

        "historical_demand":
        historical_demand,

        "safety_risk":
        safety_risk
    })

df = pd.DataFrame(data)

# =========================================================
# ENSURE NO MISSING VALUES
# =========================================================

df = df.fillna(0)

# =========================================================
# SAVE DATASET
# =========================================================

df.to_csv(
    "supply_risk_dataset_50k.csv",
    index=False
)

print(df.head())

print("\nDataset Shape:", df.shape)

print(
    "\nMissing Values:",
    df.isnull().sum().sum()
)

print(
    "\nSupply Risk Dataset Generated Successfully."
)