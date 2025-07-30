def add_elements(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return
    the sum of the elements with at most two digits from the first k elements of arr.
    Example:
        Input: arr = [111, 21, 3, 4000, 5, 6, 7, 8, 9], k = 4
        Output: 24 # sum of 21 + 3
    Constraints:
        1. 1 <= len(arr) <= 100
        2. 1 <= k <= len(arr)
    """
    total_sum = 0  # Initialize sum of valid elements
    for i in range(k):  # Iterate through the first k elements
        if -99 <= arr[i] <= 99:  # Check if the element has at most two digits
            total_sum += arr[i]  # Add to the sum if valid
    return total_sum  # Return the final sum