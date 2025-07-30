def minSubArraySum(nums):
    """
    Given an array of integers nums, find the minimum sum of any non-empty sub-array
    of nums.
    Parameters:
    nums (list): A list of integers.
    Returns:
    int: The minimum sum of any non-empty sub-array.
    Example:
    minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
    minSubArraySum([-1, -2, -3]) == -6
    """
    # Initialize the minimum sum to a large value
    min_sum = float('inf')
    current_sum = 0
    for num in nums:
        # Add the current number to the current sum
        current_sum += num
        # Update the minimum sum if the current sum is less
        if current_sum < min_sum:
            min_sum = current_sum
        # If the current sum becomes positive, reset it to 0
        if current_sum > 0:
            current_sum = 0
    return min_sum