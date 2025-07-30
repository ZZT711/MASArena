def fib4(n: int):
    """
    The Fib4 number sequence is a sequence similar to the Fibonacci sequence defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    This function computes the n-th element of the Fib4 number sequence without using recursion.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
    """
    # Handle base cases
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 0
    # Initialize an array to store the Fib4 sequence values
    fib4_sequence = [0] * (n + 1)
    # Set base cases
    fib4_sequence[0] = 0
    fib4_sequence[1] = 0
    fib4_sequence[2] = 2
    fib4_sequence[3] = 0
    # Compute the Fib4 sequence iteratively
    for i in range(4, n + 1):
        fib4_sequence[i] = (fib4_sequence[i - 1] + 
                            fib4_sequence[i - 2] + 
                            fib4_sequence[i - 3] + 
                            fib4_sequence[i - 4])
    return fib4_sequence[n]