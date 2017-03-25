# Given open, high, low and close prices return a state for a given configuration of prices
def hidden_classify(op, hi, lo, cl, cl_p, std):
    op_ret = (op-cl_p)/cl_p
    if op_ret >= -2*std and op_ret < -std:
        return 1
    elif op_ret >= -std and op_ret <= 0:
        return 2
    elif op_ret > 0 and op_ret <= std:
        return 3
    elif op_ret > std and op_ret <= 2*std:
        return 4
    elif op_ret < 2*std:
        return 5
    elif op_ret > 2*std:
        return 6

