def triangle_area(a, h):
    """
    Given length of a side and height, return area for a triangle.
    The area is calculated using the formula: Area = 0.5 * base * height.
    Parameters:
    a (float): The length of the base of the triangle.
    h (float): The height of the triangle.
    Returns:
    float: The area of the triangle.
    >>> triangle_area(5, 3)
    7.5
    >>> triangle_area(10, 2)
    10.0
    >>> triangle_area(0, 5)
    0.0
    >>> triangle_area(5, 0)
    0.0
    """
    if a < 0 or h < 0:
        raise ValueError("Base and height must be non-negative.")
    return 0.5 * a * h