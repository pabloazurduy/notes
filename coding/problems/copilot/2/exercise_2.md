
# Exercise 2: Data Cleaning and Transformation

**Difficulty:** medium
**Estimated time:** 30 minutes

## Prompt

You are given a pandas DataFrame with information about customer orders. The DataFrame has the following columns:

- `order_id`: Unique identifier for each order.
- `customer_id`: Identifier for the customer.
- `order_date`: Date of the order in `YYYY-MM-DD` format.
- `product_name`: Name of the product ordered.
- `quantity`: Number of units of the product ordered.
- `price`: Price of one unit of the product. Some values might be missing.

Your task is to write a Python function `clean_and_transform_data(df)` that takes this DataFrame as input and performs the following operations:

1. **Handle missing values:** Fill missing `price` values with the mean price of the corresponding product. If all prices for a product are missing, fill them with the mean price of all products.
2. **Calculate total price:** Create a new column `total_price` which is the product of `quantity` and `price`.
3. **Convert date:** Convert the `order_date` column to datetime objects.
4. **Extract features from date:** Create two new columns, `order_year` and `order_month`, from the `order_date` column.
5. **Aggregate data:** Group the data by `customer_id` and `order_year` and calculate the total amount spent by each customer per year. The resulting DataFrame should have columns `customer_id`, `order_year`, and `total_spent`.

The function should return the cleaned and aggregated DataFrame.

## Test Case

**Input DataFrame:**

```python
import pandas as pd
import numpy as np

data = {
    'order_id': [1, 2, 3, 4, 5, 6, 7],
    'customer_id': [101, 102, 101, 103, 102, 101, 103],
    'order_date': ['2023-01-15', '2023-01-17', '2023-02-05', '2024-03-02', '2024-03-05', '2024-04-10', '2023-01-20'],
    'product_name': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    'quantity': [2, 1, 3, 5, 2, 1, 1],
    'price': [10.0, 20.0, 10.0, np.nan, 20.0, 12.0, 30.0]
}
df = pd.DataFrame(data)
```

**Expected Output DataFrame:**

```python
   customer_id  order_year  total_spent
0          101        2023         50.0
1          101        2024         12.0
2          102        2023         20.0
3          102        2024         40.0
4          103        2023         30.0
5          103        2024        150.0
```
