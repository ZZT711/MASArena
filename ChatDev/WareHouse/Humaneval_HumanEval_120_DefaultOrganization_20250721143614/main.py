def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.
    Parameters:
    arr (list of int): The input array of integers.
    k (int): The number of maximum elements to return.
    Returns:
    list of int: A sorted list containing the maximum k numbers from arr.
    Example:
        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]
    """
    # Handle edge case where k is 0
    if k == 0:
        return []
    # Sort the array in descending order
    sorted_arr = sorted(arr, reverse=True)
    # Get the top k elements
    max_k_elements = sorted_arr[:k]
    # Return the sorted list of the maximum k elements in ascending order
    return sorted(max_k_elements)
# Example usage
if __name__ == "__main__":
    print(maximum([-3, -4, 5], 3))  # Output: [-4, -3, 5]
    print(maximum([4, -4, 4], 2))    # Output: [4, 4]
    print(maximum([-3, 2, 1, 2, -1, -2, 1], 1))  # Output: [2]