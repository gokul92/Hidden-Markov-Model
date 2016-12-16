# Given open, high, low and close prices return a state for a given configuration of prices
def ohlc_classify(op, hi, lo, cl):
    if hi < op:
        if lo < op:
            if cl < op:
                return 1
            else:
                return 2
        else:
            if cl > op:
                return 3
            else:
                return 4
    else:
        if lo > op:
            if cl > op:
                return 5
            else:
                return 6
        else:
            if lo < op:
                return 7
            else:
                return 8
