import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(400)]  # 400 days of data
store_nbrs = [1, 2]  # Two stores
families = ['GROCERY', 'BEVERAGES']  # Two product families

data = []
for store in store_nbrs:
    for family in families:
        for date in dates:
            data.append({
                'store_nbr': store,
                'family': family,
                'date': date.strftime('%Y-%m-%d'),
                'sales': np.random.randint(0, 1000)  # Random sales data
            })

df = pd.DataFrame(data)
df.to_csv('sample_input.csv', index=False)