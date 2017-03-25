from get_data_class import get_data_id
from get_states import hidden_classify
from get_obs import ohlc_classify
from get_data_eod import get_data_eod
import numpy as np
import datetime
import copy as cp
import sys


def output(x, start, end):
    for i in range(start, end + 1):
        print(x[i])
    return 0

# ********************************READ FILE AND CREATE OBJECT *****************#

# Change filename to point to file location
status = "id"

if status == "id":

    fn = "/Users/gokul/Desktop/Finance/Data/Equity/Intraday/Convert/BANKBARODA.csv"
    stock_id = get_data_id(fn, 0)
    datelist = stock_id.get_datelist()
    lookback_dates = 2
    timelength = 0
    for j in range(lookback_dates):
        date = datelist[len(datelist) - lookback_dates + j]
        timelist = stock_id.get_timelist(date)
        timelength += int(len(timelist))
    op_p = np.empty(timelength)
    hi_p = np.empty(timelength)
    lo_p = np.empty(timelength)
    cl_p = np.empty(timelength)
    last_index = 0
    for j in range(lookback_dates):
        date = datelist[len(datelist) - lookback_dates + j]
        timelist = stock_id.get_timelist(date)
        for i in range(int(len(timelist))):
            ohlc = stock_id.get_min_ohlc(date, timelist[i])
            op_p[i + last_index] = ohlc[0]
            hi_p[i + last_index] = ohlc[1]
            lo_p[i + last_index] = ohlc[2]
            cl_p[i + last_index] = ohlc[3]
            if i == int(len(timelist)) - 1:
                last_index = i + last_index + 1
    op_mean = np.mean(op_p)
    op_std_dev = np.std(op_p)
    cl_mean = np.mean(cl_p)
    cl_std_dev = np.std(cl_p)
    op_ret = op_std_dev / op_mean
    cl_ret = cl_std_dev / cl_mean
    x = np.empty(timelength - 1)
    h = np.empty(timelength - 1)
    for i in range(1, timelength):
        x[i - 1] = ohlc_classify(op_p[i], hi_p[i], lo_p[i], cl_p[i], cl_ret)
        h[i - 1] = hidden_classify(op_p[i], hi_p[i], lo_p[i], cl_p[i], cl_p[i-1], op_ret)

elif status == "eod":

    fn = "/Users/gokul/Desktop/Data/Equity/EOD/Convert/BANKBARODA.csv"
    stock_eod = get_data_eod(fn, 0)
    stock_dates = stock_eod.get_date()
    open_prices = stock_eod.get_open_price()
    high_prices = stock_eod.get_high_price()
    low_prices = stock_eod.get_low_price()
    close_prices = stock_eod.get_close_price()
    timelength = 100
    x = np.empty(timelength)
    for i in range(timelength):
        t = len(stock_dates) - timelength + i
        x[i] = ohlc_classify(open_prices[t], high_prices[t], low_prices[t], close_prices[t])

# sys.exit()
# ********************************INITIALIZATION STEP**************************#
n_data = len(x)
#print(x)
#sys.exit()

x_states = [1, 2, 3, 4, 5, 6]
h_states = [1, 2, 3, 4, 5, 6]

n_xstates = len(x_states)
n_states = len(h_states)

occurence = np.empty(n_xstates)
for i in range(n_xstates):
    temp = np.where(x == x_states[i])[0]
    occurence[i] = len(temp) / n_data

print("prior occurence ", occurence)

# Initial probability vector for hidden states - pi
pi = np.empty(n_states)
for i in range(n_states):
    # pi[i] = np.random.uniform(low=0.001, high=1.0)
    pi[i] = occurence[i]

# Normalization to ensure sum of probabilities = 1
pi = pi / np.sum(pi)

# Transition probability matrix A
A = np.empty(shape=(n_states, n_states))
for i in range(n_states):
    for j in range(n_states):
        A[i, j] = np.random.uniform(low=0.001, high=1.0)
    A[i, :] = A[i, :] / np.sum(A[i, :])

# Emission probability matrix B
B = np.empty(shape=(n_xstates, n_states))
for k in range(n_states):
    for i in range(n_xstates):
        B[i, k] = np.random.uniform(low=0.001, high=1.0)
    B[:, k] = B[:, k] / np.sum(B[:, k])
# for k in range(n_states):
#     for i in range(n_xstates):
#         B[i, k] = occurence[i]

# alpha_sc, beta_sc (scaled alpha and beta matrices), delta, epsilon and cn declaration
alpha_sc = np.empty(shape=(n_data, n_states))
beta_sc = np.empty(shape=(n_data, n_states))
cn = np.empty(n_data)
alpha_base = np.empty(n_states)
beta_base = np.empty(n_states)
delta = np.empty(shape=(n_data, n_states))
epsilon = np.empty(shape=(n_data, n_states))

iter_total = 5000

iter_no = 1

while iter_no <= iter_total:

    # ********************************EXPECTATION STEP*****************************#

    # Calculating alpha_base, cn[0] and base for alpha_sc
    alpha_base = A[0, :]

    loc = np.where(x[0] == x_states)[0]
    cn[0] = np.sum(pi * B[loc, :])

    alpha_sc[0, :] = alpha_base / cn[0]
    delta[0, :] = alpha_sc[0, :] * cn[0]

    for i in range(1, n_data):
        for j in range(n_states):
            loc = np.where(x[i] == x_states)[0]
            emis_prob = B[loc, j]
            temp = 0.0
            for k in range(n_states):
                temp += alpha_sc[i - 1, k] * A[k, j]
            delta[i, j] = emis_prob * temp
        cn[i] = np.sum(delta[i, :])
        alpha_sc[i, :] = delta[i, :] / cn[i]

    beta_base = 1
    beta_sc[-1, :] = beta_base

    # epsilon[-1,:] not initialized with values. Thus do not use it anywhere !
    for i in range(n_data - 2, -1, -1):
        for j in range(n_states):
            temp = 0.0
            for k in range(n_states):
                loc = np.where(x[i + 1] == x_states)[0]
                emis_prob = B[loc, k]
                temp += beta_sc[i + 1, k] * emis_prob * A[j, k]
            epsilon[i, j] = temp
        beta_sc[i, :] = epsilon[i, :] / cn[i + 1]

    # ********************************MAXIMIZATION STEP****************************#

    # Initial probability vector pi
    pi = alpha_sc[0, :] * beta_sc[0, :] / np.sum(alpha_sc[0, :] * beta_sc[0, :])

    # Transition probability matrix A
    Atemp = np.empty_like(A)
    for j in range(n_states):
        for k in range(n_states):
            numerator = 0
            for n in range(1, n_data):
                loc = np.where(x[n] == x_states)[0]
                emis_prob = B[loc, k]
                numerator += alpha_sc[n - 1, j] * beta_sc[n, k] * emis_prob * A[j, k] / cn[n]
            denominator = 0
            for l in range(n_states):
                temp = 0
                for n in range(1, n_data):
                    loc = np.where(x[n] == x_states)[0]
                    emis_prob = B[loc, l]
                    temp += alpha_sc[n - 1, j] * beta_sc[n, l] * emis_prob * A[j, l] / cn[n]
                denominator += temp
            Atemp[j, k] = numerator / denominator
    A[:] = Atemp

    # Emission probability matrix B
    for i in range(n_xstates):
        arg_list = np.where(x == x_states[i])[0]
        for k in range(n_states):
            numerator = 0
            for n in range(n_data):
                if x[n] == x_states[i]:
                    numerator += alpha_sc[n, k] * beta_sc[n, k]
            denominator = np.sum(np.array([alpha_sc[t, k] * beta_sc[t, k] for t in range(n_data)]))
            B[i, k] = numerator / denominator

    print(iter_no)
    if iter_no % 200 == 0:
        print(A)
    iter_no = iter_no + 1

# Given the learned parameters above, predicting the probable states for the next observation n+1
# ********************************PREDICTION STEP****************************#


# Calculating probability of given set of observations X = (x1, x2, x3, ... xn)
px = np.prod(cn)

pnext = np.zeros(n_xstates)

# Calculating alpha as it is used to predict the next state given X.
alpha = np.empty(shape=(n_data, n_states))
for i in range(n_data):
    ctemp = 1
    for j in range(i + 1):
        ctemp = ctemp * cn[j]
    alpha[i, :] = alpha_sc[i, :] * ctemp

for k in range(n_xstates):
    for i in range(n_states):
        temp_sum = 0
        for j in range(n_states):
            temp_sum += alpha[-1, j] * A[j, i]
        # print(temp_sum, B[k, i], px)
        pnext[k] += temp_sum * B[k, i] / px

print("probability of next trade taking the two possible states are ", pnext)
print("sum of probabilities", np.sum(pnext))
