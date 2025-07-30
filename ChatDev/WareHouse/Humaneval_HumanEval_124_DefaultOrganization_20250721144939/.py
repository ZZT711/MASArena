def valid_date(date):
    """
    Validates a given date string and returns True if the date is valid,
    otherwise returns False. The date is valid if all of the following rules are satisfied:
    1. The date string is not empty.
    2. The number of days is not less than 1 or higher than 31 days for months 1,3,5,7,8,10,12.
       The number of days is not less than 1 or higher than 30 days for months 4,6,9,11.
       The number of days is not less than 1 or higher than 29 for the month 2.
    3. The months should not be less than 1 or higher than 12.
    4. The date should be in the format: mm-dd-yyyy
    """
    # Check if the date string is empty
    if not date:
        return False
    # Check if the date format is correct
    if len(date) != 10 or date[2] != '-' or date[5] != '-':
        return False
    # Split the date into components
    try:
        month, day, year = map(int, date.split('-'))
    except ValueError:
        return False
    # Validate the month
    if month < 1 or month > 12:
        return False
    # Validate the day based on the month
    if month in [1, 3, 5, 7, 8, 10, 12]:  # 31 days
        if day < 1 or day > 31:
            return False
    elif month in [4, 6, 9, 11]:  # 30 days
        if day < 1 or day > 30:
            return False
    elif month == 2:  # February
        if day < 1 or day > 29:
            return False
    return True