'''
Return list of all prefixes from shortest to longest of the input string.
Parameters:
string (str): The input string from which prefixes will be generated.
Returns:
List[str]: A list containing all prefixes of the input string.
Examples:
>>> all_prefixes('abc')
['a', 'ab', 'abc']
>>> all_prefixes('')
[]
>>> all_prefixes('a')
['a']
>>> all_prefixes('hello')
['h', 'he', 'hel', 'hell', 'hello']
'''
def all_prefixes(string: str) -> List[str]:
    # Initialize an empty list to store prefixes
    prefixes = []
    # Loop through the string to generate prefixes
    for i in range(1, len(string) + 1):
        prefixes.append(string[:i])
    return prefixes