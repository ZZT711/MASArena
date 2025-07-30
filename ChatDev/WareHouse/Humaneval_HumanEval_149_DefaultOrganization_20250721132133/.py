'''
This function accepts a list of strings, removes strings with odd lengths,
and returns a sorted list of the remaining strings. The sorting is done first
by the length of the strings and then alphabetically for strings of the same length.
'''
def sorted_list_sum(lst):
    # Filter out strings with odd lengths
    filtered_list = [s for s in lst if len(s) % 2 == 0]
    # Sort the filtered list first by length and then alphabetically
    sorted_list = sorted(filtered_list, key=lambda x: (len(x), x))
    return sorted_list
# Example assertions to test the function
assert sorted_list_sum(["aa", "a", "aaa"]) == ["aa"]
assert sorted_list_sum(["ab", "a", "aaa", "cd"]) == ["ab", "cd"]