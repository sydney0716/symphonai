def parse_count(value):
    try:
        return int(value)
    except ValueError:
        return 0
