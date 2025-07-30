def file_name_check(file_name):
    """
    Create a function which takes a string representing a file's name, and returns
    'Yes' if the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions 
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'.
    - The substring before the dot should not be empty, and it starts with a letter from 
      the Latin alphabet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll'].
    """
    def has_valid_digits(name):
        """Check if the file name has no more than three digits."""
        digit_count = sum(c.isdigit() for c in name)
        return digit_count <= 3
    def has_exactly_one_dot(name):
        """Check if the file name contains exactly one dot."""
        return name.count('.') == 1
    def is_valid_prefix(name):
        """Check if the substring before the dot is valid."""
        prefix, _ = name.split('.', 1)
        return len(prefix) > 0 and prefix[0].isalpha()
    def is_valid_suffix(name):
        """Check if the substring after the dot is valid."""
        _, suffix = name.split('.', 1)
        return suffix in ['txt', 'exe', 'dll']
    # Check all conditions
    if (has_valid_digits(file_name) and 
        has_exactly_one_dot(file_name) and 
        is_valid_prefix(file_name) and 
        is_valid_suffix(file_name)):
        return 'Yes'
    else:
        return 'No'