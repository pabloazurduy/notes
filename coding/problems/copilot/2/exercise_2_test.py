
import unittest
import pandas as pd
import numpy as np
from exercise_2 import clean_and_transform_data

class TestCleanAndTransformData(unittest.TestCase):

    def test_clean_and_transform_data(self):
        # Input DataFrame from the exercise
        data = {
            'order_id': [1, 2, 3, 4, 5, 6, 7],
            'customer_id': [101, 102, 101, 103, 102, 101, 103],
            'order_date': ['2023-01-15', '2023-01-17', '2023-02-05', '2024-03-02', '2024-03-05', '2024-04-10', '2023-01-20'],
            'product_name': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
            'quantity': [2, 1, 3, 5, 2, 1, 1],
            'price': [10.0, 20.0, 10.0, np.nan, 20.0, 12.0, 30.0]
        }
        df = pd.DataFrame(data)

        # Expected output DataFrame from the exercise
        expected_data = {
            'customer_id': [101, 101, 102, 102, 103, 103],
            'order_year': [2023, 2024, 2023, 2024, 2023, 2024],
            'total_spent': [50.0, 12.0, 20.0, 40.0, 30.0, 150.0]
        }
        expected_df = pd.DataFrame(expected_data)

        # Run the function
        result_df = clean_and_transform_data(df.copy())

        # Sort both dataframes to ensure comparison is correct
        result_df = result_df.sort_values(by=['customer_id', 'order_year']).reset_index(drop=True)
        expected_df = expected_df.sort_values(by=['customer_id', 'order_year']).reset_index(drop=True)

        # Check if the result is as expected
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_all_prices_missing_for_a_product(self):
        # Test case where a product has all its prices missing
        data = {
            'order_id': [1, 2, 3, 4],
            'customer_id': [101, 102, 101, 102],
            'order_date': ['2023-01-15', '2023-01-17', '2023-02-05', '2023-02-10'],
            'product_name': ['A', 'B', 'A', 'B'],
            'quantity': [2, 1, 3, 2],
            'price': [10.0, np.nan, 10.0, np.nan]
        }
        df = pd.DataFrame(data)
        
        # The mean of all prices is 10.0. So NaN for product B should be 10.0
        expected_data = {
            'customer_id': [101, 102],
            'order_year': [2023, 2023],
            'total_spent': [50.0, 30.0] # 1*10 + 2*10 for customer 102
        }
        expected_df = pd.DataFrame(expected_data)
        
        result_df = clean_and_transform_data(df.copy())
        
        result_df = result_df.sort_values(by=['customer_id', 'order_year']).reset_index(drop=True)
        expected_df = expected_df.sort_values(by=['customer_id', 'order_year']).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
