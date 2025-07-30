def is_bored(S):
    """
    Count the number of sentences that start with the word 'I'.
    A sentence is defined as a sequence of characters ending with '.', '?' or '!'.
    Args:
        S (str): The input string containing multiple sentences.
    Returns:
        int: The count of sentences that start with 'I'.
    Examples:
        >>> is_bored("Hello world")
        0
        >>> is_bored("The sky is blue. The sun is shining. I love this weather")
        1
    """
    # Split the input string into sentences based on the delimiters
    sentences = []
    current_sentence = []
    for char in S:
        if char in '.!?':
            current_sentence.append(char)
            sentences.append(''.join(current_sentence).strip())
            current_sentence = []
        else:
            current_sentence.append(char)
    # Add the last sentence if it exists
    if current_sentence:
        sentences.append(''.join(current_sentence).strip())
    # Count the number of sentences that start with 'I'
    count = sum(1 for sentence in sentences if sentence.startswith('I'))
    return count