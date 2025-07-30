'''
This module provides a function to calculate the area of a triangle given the lengths of its three sides.
The function checks if the sides form a valid triangle and returns the area rounded to two decimal points,
or -1 if the sides do not form a valid triangle.
'''
def triangle_area(a, b, c):
    '''
    Given the lengths of the three sides of a triangle. Return the area of
    the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
    Otherwise return -1.
    Three sides make a valid triangle when the sum of any two sides is greater 
    than the third side.
    Parameters:
    a (float): Length of side a
    b (float): Length of side b
    c (float): Length of side c
    Returns:
    float: Area of the triangle rounded to 2 decimal points or -1 if invalid.
    '''
    # Check for valid triangle condition
    if a + b > c and a + c > b and b + c > a:
        # Calculate the semi-perimeter
        s = (a + b + c) / 2
        # Calculate the area using Heron's formula
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        # Return the area rounded to 2 decimal points
        return round(area, 2)
    else:
        # Return -1 if the sides do not form a valid triangle
        return -1