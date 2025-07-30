def is_prime(n):
    """
    Check if a number n is a prime number.
    A prime number is greater than 1 and has no divisors other than 1 and itself.
    Args:
        n (int): The number to check for primality.
    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
def words_in_sentence(sentence):
    """
    Given a string representing a sentence, return a string that contains the words
    from the original sentence whose lengths are prime numbers.
    The order of the words in the new string should be the same as the original one.
    Args:
        sentence (str): The input sentence containing words separated by spaces.
    Returns:
        str: A string containing words with prime lengths.
    """
    words = sentence.split()
    prime_length_words = [word for word in words if is_prime(len(word))]
    return ' '.join(prime_length_words)
# Example usage (this part can be commented out or removed in production):
if __name__ == "__main__":
    print(words_in_sentence("This is a test"))  # Output: "is"
    print(words_in_sentence("lets go for swimming"))  # Output: "go for"