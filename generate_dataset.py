import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

rows = 50000
start_date = datetime(2023, 1, 1)

plants = ["Plant_A", "Plant_B", "Plant_C", "Plant_D"]
regions = ["North", "South", "East", "West"]
suppliers = ["Supplier_X", "Supplier_Y", "Supplier_Z", "Supplier_M"]
machine_statuses = ["Operational", "Maintenance", "Failure"]

data = []

for i in range(rows):

    production_volume = np.random.randint(100, 1000)
    inventory_level = np.random.randint(50, 5000)
    machine_efficiency = round(np.random.uniform(60, 99), 2)
    downtime_hours = round(np.random.uniform(0, 20), 2)
    energy_consumption = round(np.random.uniform(100, 2000), 2)
    defect_rate = round(np.random.uniform(0, 10), 2)
    supplier_delay_days = np.random.randint(0, 15)
    employee_count = np.random.randint(20, 500)
    market_demand_index = round(np.random.uniform(50, 150), 2)

    raw_material_cost = round(
        production_volume * np.random.uniform(2.0, 5.0)
        + energy_consumption * 0.3
        + supplier_delay_days * 25,
        2
    )

    transportation_cost = round(
        production_volume * np.random.uniform(0.5, 2.0),
        2
    )

    maintenance_cost = round(
        downtime_hours * np.random.uniform(50, 300),
        2
    )

    procurement_cost = round(
        raw_material_cost
        + transportation_cost
        + maintenance_cost,
        2
    )

    historical_demand = round(
        production_volume * 1.5
        + market_demand_index * 12
        - downtime_hours * 18
        - defect_rate * 25
        + np.random.normal(0, 100),
        2
    )

    risk_score = (
        defect_rate * 2
        + downtime_hours * 1.5
        + supplier_delay_days * 1.2
        - machine_efficiency * 0.5
    )

    if risk_score > 35:
        safety_risk = "High"
    elif risk_score > 20:
        safety_risk = "Medium"
    else:
        safety_risk = "Low"

    data.append({

        "date": (
            start_date + timedelta(days=i % 730)
        ).strftime("%Y-%m-%d"),

        "plant_location": random.choice(plants),

        "region": random.choice(regions),

        "supplier_name": random.choice(suppliers),

        "machine_status": random.choice(machine_statuses),

        "production_volume": production_volume,

        "inventory_level": inventory_level,

        "machine_efficiency": machine_efficiency,

        "downtime_hours": downtime_hours,

        "energy_consumption": energy_consumption,

        "defect_rate": defect_rate,

        "supplier_delay_days": supplier_delay_days,

        "employee_count": employee_count,

        "market_demand_index": market_demand_index,

        "raw_material_cost": raw_material_cost,

        "transportation_cost": transportation_cost,

        "maintenance_cost": maintenance_cost,

        "procurement_cost": procurement_cost,

        "historical_demand": historical_demand,

        "safety_risk": safety_risk
    })

df = pd.DataFrame(data)

df.to_csv(
    "full_manufacturing_dataset_50k.csv",
    index=False
)

print(df.head())
print(df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Dataset generated successfully.")