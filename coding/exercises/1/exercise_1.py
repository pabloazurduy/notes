import numpy as np

def rolling_statistics(data: list, window_size: int) -> list:
    """
    Calculate rolling mean and standard deviation.
    
    Args:
        data: List of numeric values
        window_size: Size of the rolling window
        
    Returns:
        List of tuples (mean, std) for each window position
    """
    # TODO: Implement the solution

    windows = [ data[i:i+window_size] for i in range(len(data)-window_size)]

    rolling_mean = [np.mean(w) for w in windows]
    rolling_std = [np.round(np.std(w),0) for w in windows]

    return list(zip(rolling_mean, rolling_std))


def rolling_statistics_solution(data: list, window_size: int) -> list:
    """
    Solution implementation for rolling statistics.
    """
    if window_size > len(data) or window_size <= 0:
        return []
    
    results = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        
        # Calculate mean
        mean = sum(window) / len(window)
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance ** 0.5
        
        # Round to 2 decimal places
        results.append((round(mean, 2), round(std, 2)))
    
    return results
