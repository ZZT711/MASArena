def parse_music(music_string: str) -> List[int]:
    """
    Input to this function is a string representing musical notes in a special ASCII format.
    The function parses this string and returns a list of integers corresponding to how many beats
    each note lasts.
    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quarter note, lasts one beat
    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    # Initialize an empty list to store the beats
    beats = []
    # Split the input string into individual notes
    notes = music_string.split()
    # Map each note to its corresponding beat value
    for note in notes:
        if note == 'o':
            beats.append(4)  # Whole note
        elif note == 'o|':
            beats.append(2)  # Half note
        elif note == '.|':
            beats.append(1)  # Quarter note
        else:
            raise ValueError(f"Invalid note: {note}")  # Handle unexpected notes
    return beats