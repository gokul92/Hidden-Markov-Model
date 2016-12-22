# Given open, high, low and close prices return a state for a given configuration of prices
def ohlc_classify(op, hi, lo, cl, std):
    cl_ret = (cl-op)/op
    if cl_ret >= -2*std and cl_ret < -std:
        return 1
    elif cl_ret >= -std and cl_ret <= 0:
        return 2
    elif cl_ret > 0 and cl_ret <= std:
        return 3
    elif cl_ret > std and cl_ret <= 2*std:
        return 4
    elif cl_ret < 2*std:
        return 5
    elif cl_ret > 2*std:
        return 6

