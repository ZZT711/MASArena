def encode(message):
    """
    Write a function that takes a message, and encodes in such a 
    way that it swaps case of all letters, replaces all vowels in 
    the message with the letter that appears 2 places ahead of that 
    vowel in the english alphabet. 
    Assume only letters. 
    Examples:
    >>> encode('test')
    'TGST'
    >>> encode('This is a message')
    'tHKS KS C MGSSCGG'
    """
    def swap_case_and_replace_vowels(char):
        """
        Helper function to swap case and replace vowels.
        Args:
        char (str): A single character from the message.
        Returns:
        str: The modified character after case swap and vowel replacement.
        """
        vowels = 'aeiouAEIOU'
        if char in vowels:
            # Replace vowel with the letter that appears 2 places ahead
            if char.islower():
                return chr(ord(char) + 2).upper()
            else:
                return chr(ord(char) + 2).lower()
        else:
            # Swap case for consonants
            if char.islower():
                return char.upper()
            else:
                return char.lower()
    # Process each character in the message and join the results
    encoded_message = ''.join(swap_case_and_replace_vowels(char) for char in message)
    return encoded_message