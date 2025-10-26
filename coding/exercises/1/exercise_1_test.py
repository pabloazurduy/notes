from exercise_1 import rolling_statistics, rolling_statistics_solution


def test_basic_functionality():
    """Test Case 1: Basic functionality"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_size = 3
    expected = [
        (2.0, 1.0),
        (3.0, 1.0),
        (4.0, 1.0),
        (5.0, 1.0),
        (6.0, 1.0),
        (7.0, 1.0),
        (8.0, 1.0),
        (9.0, 1.0)
    ]
    result = rolling_statistics(data, window_size)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test Case 1 passed: Basic functionality")


def test_window_equals_data_length():
    """Test Case 2: Window size equals data length"""
    data = [5, 10, 15, 20]
    window_size = 4
    expected = [(12.5, 6.45)]
    result = rolling_statistics(data, window_size)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test Case 2 passed: Window size equals data length")


def test_window_larger_than_data():
    """Test Case 3: Window size larger than data"""
    data = [1, 2, 3]
    window_size = 5
    expected = []
    result = rolling_statistics(data, window_size)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test Case 3 passed: Window size larger than data")


def test_single_element_windows():
    """Test Case 4: Single element windows"""
    data = [10, 20, 30, 40]
    window_size = 1
    expected = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
    result = rolling_statistics(data, window_size)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test Case 4 passed: Single element windows")


def run_all_tests():
    """Run all test cases"""
    print("Running tests for rolling_statistics()...\n")
    
    try:
        test_basic_functionality()
        test_window_equals_data_length()
        test_window_larger_than_data()
        test_single_element_windows()
        print("\n✓ All tests passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error running tests: {e}")


if __name__ == "__main__":
    # First, let's verify the solution works
    print("Testing solution implementation:\n")
    from exercise_1 import rolling_statistics_solution as test_func
    
    # Temporarily replace the function for testing
    import exercise_1
    original_func = exercise_1.rolling_statistics
    exercise_1.rolling_statistics = test_func
    
    run_all_tests()
    
    print("\n" + "="*50)
    print("Now implement your solution in exercise_1.py")
    print("and run this test file again!")
