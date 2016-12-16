from get_data_class import get_data_id
from get_states import ohlc_classify
import numpy as np

# ********************************READ FILE AND CREATE OBJECT *****************#

# Change filename to point to file location
fn = "/Users/gokul/Desktop/Finance/NIFTY/Intraday/india/NIFTY 50.csv"
nifty_id = get_data_id(fn, 0)

datelist = nifty_id.get_datelist()

date = datelist[50]
timelist = nifty_id.get_timelist(date)
if len(timelist) % 2 == 0:
    timelength = len(timelist) / 2
else:
    timelength = (len(timelist) - 1) / 2
timelength = int(timelength)
x = np.empty(timelength)
for i in range(timelength):
    ohlc = nifty_id.get_min_ohlc(date, timelist[i])
    x[i] = ohlc_classify(ohlc[0], ohlc[1], ohlc[2], ohlc[3])

# ********************************INITIALIZATION STEP**************************#
n_data = len(x)
n_states = 2

x_states = [7, 8]
n_xstates = len(x_states)

occurence = np.empty(n_xstates)
for i in range(n_xstates):
    temp = np.where(x == x_states[i])[0]
    occurence[i] = len(temp) / n_data

# Initial probability vector for hidden states - pi
pi = np.empty(n_states)
for i in range(n_states):
    pi[i] = np.random.uniform(low=0.001, high=1.0)

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
        B[i, k] = occurence[i]

# alpha_sc, beta_sc (scaled alpha and beta matrices), delta, epsilon and cn declaration
alpha_sc = np.empty(shape=(n_data, n_states))
beta_sc = np.empty(shape=(n_data, n_states))
cn = np.empty(n_data)
alpha_base = np.empty(n_states)
beta_base = np.empty(n_states)
delta = np.empty(shape=(n_data, n_states))
epsilon = np.empty(shape=(n_data, n_states))

iter_total = 100

iter_no = 1

while iter_no <= iter_total:

    # ********************************EXPECTATION STEP*****************************#

    # Calculating alpha_base, cn[0] and base for alpha_sc
    alpha_base = A[0, :]
    cn[0] = np.sum(pi * B[0, :])
    alpha_sc[0, :] = alpha_base / cn[0]
    delta[0, :] = alpha_sc[0, :] * cn[0]

    for i in range(1, n_data):
        for j in range(n_states):
            loc = np.where(x[i] == x_states)[0]
            emis_prob = B[loc, j]
            delta[i, j] = emis_prob * np.sum(alpha_sc[i - 1, :] * A[:, j])
        cn[i] = np.sum(delta[i, :])
        alpha_sc[i, :] = delta[i, :] / cn[i]

    beta_base = 1
    beta_sc[-1, :] = beta_base

    # epsilon[-1,:] not initialized with values. Thus do not use it anywhere !
    for i in range(n_data - 2, -1, -1):
        for j in range(n_states):
            loc = np.where(x[i + 1] == x_states)[0]
            emis_prob = B[loc, j]
            epsilon[i, j] = np.sum(beta_sc[i + 1, :] * emis_prob * A[j, :])
        beta_sc[i, :] = epsilon[i, :] / cn[i + 1]

    # ********************************MAXIMIZATION STEP****************************#

    # Initial probability vector pi
    pi = alpha_sc[0, :] * beta_sc[0, :] / np.sum(alpha_sc[0, :] * beta_sc[0, :])

    # Transition probability matrix A
    Atemp = np.empty_like(A)
    for i in range(n_states):
        for j in range(n_states):
            numerator = 0
            denominator = 0
            for n in range(0, n_data - 1):
                loc = np.where(x[n] == x_states)[0]
                emis_prob = B[loc, i]
                numerator = numerator + alpha_sc[n - 1, i] * beta_sc[n, j] * emis_prob * A[i, j] / cn[n]
            for l in range(n_states):
                temp = 0
                for n in range(0, n_data - 1):
                    loc = np.where(x[n] == x_states)[0]
                    emis_prob = B[loc, l]
                    temp = temp + alpha_sc[n - 1, i] * beta_sc[n, l] * emis_prob * A[i, l] / cn[n]
                denominator += temp
            Atemp[i, j] = numerator / denominator

    A[:] = Atemp

    # Emission probability matrix B
    for i in range(n_xstates):
        arg_list = np.where(x == x_states[i])[0]
        for k in range(n_states):
            numerator = 0
            for n in range(n_data):
                if x[n] == x_states[i]:
                    numerator = numerator + alpha_sc[n, k] * beta_sc[n, k]
            denominator = np.sum(np.array([alpha_sc[t, k] * beta_sc[t, k] for t in range(n_data)]))
            B[i, k] = numerator / denominator

    print(iter_no)
    iter_no = iter_no + 1

# Given the learned parameters above, predicting the probable states for the next observation
# ********************************PREDICTION STEP****************************#

# Calculating alpha as it is used in the prediction the next set of states.
alpha = np.empty(shape=(n_data, n_states))
for i in range(n_data):
    ctemp = 1
    for j in range(i + 1):
        ctemp = ctemp * cn[j]
    alpha[i, :] = alpha_sc[i, :] * ctemp

# Calculating probability of given set of observations X = (x1, x2, x3, ... xn)
px = np.sum(alpha[-1, :])

pnext = np.zeros(n_xstates)

for k in range(n_xstates):
    for i in range(n_states):
        temp_sum = 0
        for j in range(n_states):
            temp_sum = temp_sum + alpha[-1, j] * A[i, j]
        pnext[k] += temp_sum * B[k, i] / px

print("probability of next trade taking the two possible states are ", pnext)
