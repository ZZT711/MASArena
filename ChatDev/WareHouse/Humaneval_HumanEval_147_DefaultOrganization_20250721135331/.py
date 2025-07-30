'''
This function calculates the number of valid triples (a[i], a[j], a[k]) from an array
generated based on the formula a[i] = i * i - i + 1, where the sum of the triples is a 
multiple of 3. The function takes a positive integer n as input and returns the count 
of such triples.
Parameters:
n (int): A positive integer representing the length of the array.
Returns:
int: The number of valid triples.
'''
def get_max_triples(n):
    # Generate the array a based on the given formula
    a = [i * i - i + 1 for i in range(1, n + 1)]
    count = 0
    # Iterate through all possible triples (i, j, k) with i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check if the sum of the triple is a multiple of 3
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    return count