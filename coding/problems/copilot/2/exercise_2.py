
import pandas as pd
import numpy as np

def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms the customer order data.

    Args:
        df: DataFrame with customer order data.

    Returns:
        A cleaned and aggregated DataFrame.
    """
    df_clean = df.copy()
    #1. missing values 

    df_clean['price'] = df['price'].fillna(df['price'].mean())
    df_clean['total_price'] = df['price'] * df['quantity']
    df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])
    df_clean['order_month'] = df_clean['order_date'].dt.month
    df_clean['order_year'] = df_clean['order_date'].dt.year.astype('int64')


    df_agg=df_clean.groupby(by=['customer_id', 'order_year'], as_index=False).agg({'total_price':'sum'})
    df_agg.rename(columns={'total_price':'total_spent'}, inplace=True)
    return df_agg


if __name__ == '__main__':
    # Example usage with the test case from the exercise description
    data = {
        'order_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'customer_id': [101, 102, 101, 103, 102, 101, 103, 104],
        'order_date': ['2023-01-15', '2023-01-17', '2023-02-05', '2024-03-02', '2024-03-05', '2024-04-10', '2023-01-20', '2023-05-01'],
        'product_name': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'D'],
        'quantity': [2, 1, 3, 5, 2, 1, 1, 4],
        'price': [10.0, 20.0, 10.0, np.nan, 20.0, 12.0, 30.0, np.nan]
    }
    df = pd.DataFrame(data)
    print(df)

    # The mean price for product C is 30.0. The NaN for product C will be filled with 30.0
    # The price for product D is all NaN, so it will be filled with the global mean of non-NaN prices.
    # (10.0 + 20.0 + 10.0 + 20.0 + 12.0 + 30.0) / 6 = 102.0 / 6 = 17.0
    # So the NaN for product D will be filled with 17.0

    cleaned_df = clean_and_transform_data(df.copy())
    print("Cleaned and Transformed DataFrame:")
    print(cleaned_df)

    # Expected output for the provided example:
    #    customer_id  order_year  total_spent
    # 0          101        2023         50.0
    # 1          101        2024         12.0
    # 2          102        2023         20.0
    # 3          102        2024         40.0
    # 4          103        2023         30.0
    # 5          103        2024        150.0
    # 6          104        2023         68.0
