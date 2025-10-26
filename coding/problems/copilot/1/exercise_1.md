# Exercise 1: Rolling Window Statistics

## Problem Statement

Given a list of numeric values and a window size `k`, compute the rolling mean and rolling standard deviation for each window. Return a list of tuples where each tuple contains `(rolling_mean, rolling_std)` for that window position.

## Requirements

- If the window size is larger than the list length, return an empty list
- Use at least `k` elements for each window (don't compute statistics for incomplete windows at the start)
- Round results to 2 decimal places

## Function Signature

```python
def rolling_statistics(data: list, window_size: int) -> list:
    """
    Calculate rolling mean and standard deviation.
    
    Args:
        data: List of numeric values
        window_size: Size of the rolling window
        
    Returns:
        List of tuples (mean, std) for each window position
    """
    pass
```

## Test Cases

### Test Case 1: Basic functionality
```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
window_size = 3
expected = [
    (2.0, 1.0),   # window [1,2,3]
    (3.0, 1.0),   # window [2,3,4]
    (4.0, 1.0),   # window [3,4,5]
    (5.0, 1.0),   # window [4,5,6]
    (6.0, 1.0),   # window [5,6,7]
    (7.0, 1.0),   # window [6,7,8]
    (8.0, 1.0),   # window [7,8,9]
    (9.0, 1.0)    # window [8,9,10]
]
```

### Test Case 2: Window size equals data length
```python
data = [5, 10, 15, 20]
window_size = 4
expected = [(12.5, 6.45)]
```

### Test Case 3: Window size larger than data
```python
data = [1, 2, 3]
window_size = 5
expected = []
```

### Test Case 4: Single element windows
```python
data = [10, 20, 30, 40]
window_size = 1
expected = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
```

## Skills Tested

- Data manipulation with lists
- Statistical computations (mean, standard deviation)
- Edge case handling
- Working with sliding windows
