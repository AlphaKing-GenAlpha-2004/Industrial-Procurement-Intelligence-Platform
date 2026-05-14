import pandas as pd
import random
from faker import Faker
import os

fake = Faker()

os.makedirs("data/raw", exist_ok=True)

rows = 5000

data = []

for _ in range(rows):

    data.append({

        "historical_demand":
            random.randint(50, 1000),

        "inventory_level":
            random.randint(100, 5000),

        "raw_material_cost":
            round(random.uniform(100, 1000), 2),

        "machine_utilization":
            round(random.uniform(50, 100), 2),

        "supplier_rating":
            round(random.uniform(1, 5), 2),

        "lead_time":
            random.randint(1, 30),

        "temperature":
            round(random.uniform(20, 120), 2),

        "pressure":
            round(random.uniform(10, 300), 2),

        "labor_hours":
            round(random.uniform(5, 50), 2),

        "production_cost":
            round(random.uniform(1000, 10000), 2),

        "procurement_source":
            random.choice([
                "InHouse",
                "External"
            ]),

        "safety_risk":
            random.choice([
                "Low",
                "Medium",
                "High"
            ])
    })

df = pd.DataFrame(data)

df.to_csv(
    "data/raw/manufacturing_data.csv",
    index=False
)

print("\nDataset generated successfully!")
print(df.head())