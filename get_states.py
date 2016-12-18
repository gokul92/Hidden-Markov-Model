# Given open, high, low and close prices return a state for a given configuration of prices
def ohlc_classify(op, hi, lo, cl):
    if cl <= op:
        return 7
    elif cl > op:
        return 8
