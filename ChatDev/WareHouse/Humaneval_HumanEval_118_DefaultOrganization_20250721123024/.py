def get_closest_vowel(word):
    """
    You are given a word. Your task is to find the closest vowel that stands between 
    two consonants from the right side of the word (case sensitive).
    Vowels in the beginning and ending don't count. Return empty string if you didn't
    find any vowel that meets the above condition. 
    You may assume that the given string contains English letters only.
    Example:
    get_closest_vowel("yogurt") ==> "u"
    get_closest_vowel("FULL") ==> "U"
    get_closest_vowel("quick") ==> ""
    get_closest_vowel("ab") ==> ""
    """
    vowels = "aeiouAEIOU"  # Define vowels (case sensitive)
    length = len(word)
    # Edge case: if the word is too short to have consonants surrounding a vowel
    if length < 3:
        return ""
    # Iterate from the end of the word to the beginning
    for i in range(length - 2, 0, -1):
        if word[i] in vowels:  # Check if the current character is a vowel
            # Check if it is surrounded by consonants
            if word[i - 1].isalpha() and word[i + 1].isalpha() and \
               word[i - 1] not in vowels and word[i + 1] not in vowels:
                return word[i]  # Return the vowel if conditions are met
    return ""  # Return empty string if no valid vowel is found