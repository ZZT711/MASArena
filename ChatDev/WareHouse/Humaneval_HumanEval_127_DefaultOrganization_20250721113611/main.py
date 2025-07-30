'''
Main entry point of the application to determine the intersection of intervals.
'''
from interval import intersection
def main():
    # Example intervals to test the function
    test_cases = [
        ((1, 2), (2, 3)),
        ((-1, 1), (0, 4)),
        ((-3, -1), (-5, 5))
    ]
    for interval1, interval2 in test_cases:
        result = intersection(interval1, interval2)
        print(f"intersection({interval1}, {interval2}) ==> {result}")
if __name__ == "__main__":
    main()