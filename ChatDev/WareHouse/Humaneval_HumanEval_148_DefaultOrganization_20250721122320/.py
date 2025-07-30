def bf(planet1, planet2):
    '''
    There are eight planets in our solar system: the closest to the Sun 
    is Mercury, the next one is Venus, then Earth, Mars, Jupiter, Saturn, 
    Uranus, Neptune.
    Write a function that takes two planet names as strings planet1 and planet2. 
    The function should return a tuple containing all planets whose orbits are 
    located between the orbit of planet1 and the orbit of planet2, sorted by 
    the proximity to the sun. 
    The function should return an empty tuple if planet1 or planet2
    are not correct planet names. 
    Examples
    bf("Jupiter", "Neptune") ==> ("Saturn", "Uranus")
    bf("Earth", "Mercury") ==> ("Venus")
    bf("Mercury", "Uranus") ==> ("Venus", "Earth", "Mars", "Jupiter", "Saturn")
    '''
    # List of planets in order of proximity to the Sun
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    # Validate if the planet names are correct
    if planet1 not in planets or planet2 not in planets:
        return ()
    # Get the indices of the planets
    index1 = planets.index(planet1)
    index2 = planets.index(planet2)
    # Determine the range of indices
    start_index = min(index1, index2) + 1
    end_index = max(index1, index2)
    # Return the tuple of planets in between
    return tuple(planets[start_index:end_index])
# Example usage
print(bf("Jupiter", "Neptune"))  # Output: ("Saturn", "Uranus")
print(bf("Earth", "Mercury"))    # Output: ("Venus")
print(bf("Mercury", "Uranus"))   # Output: ("Venus", "Earth", "Mars", "Jupiter", "Saturn")