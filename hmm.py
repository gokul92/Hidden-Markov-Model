from get_data_class import get_data_id
from get_states import ohlc_classify
import numpy as np

#********************************READ FILE AND CREATE OBJECT *****************#
fn = "/Users/gokul/Desktop/Finance/NIFTY/Intraday/india/NIFTY 50.csv"
nifty_id = get_data_id(fn, 0)

datelist = nifty_id.get_datelist()

date = datelist[0]
timelist = nifty_id.get_timelist(date)
x = np.empty(len(timelist))
for i in range(len(timelist)):
    ohlc = nifty_id.get_min_ohlc(date, timelist[i])
    x[i] = ohlc_classify(ohlc[0], ohlc[1], ohlc[2], ohlc[3])

#********************************INITIALIZATION STEP**************************#
n_data = len(x)
n_states = 3

x_states = [7, 8]
n_xstates = len(x_states)

# Initial probability vector pi
pi = np.empty(n_states)
for i in range(n_states):
    pi[i] = np.random.uniform(low=0.001, high=1.0)
    
# Normalization to ensure sum of probabilities = 1
pi = pi/np.sum(pi)

# Transition probability matrix A
A = np.empty(shape=(n_states, n_states))
for i in range(n_states):
    for j in range(n_states):
        A[i,j] = np.random.uniform(low=0.001, high=1.0)
    A[i,:] = A[i,:]/np.sum(A[i,:])
    
# Emission probability matrix B
B = np.empty(shape=(n_xstates, n_states))
for i in range(n_xstates):
    for k in range(n_states):
        B[i,k] = np.random.uniform(low=0.001, high=1.0)
    B[i,:] = B[i,:]/np.sum(B[i,:])
    
# alpha_sc, beta_sc (scaled alpha and beta matrices), delta, epsilon and cn declaration
alpha_sc = np.empty(shape=(n_data, n_states))
beta_sc = np.empty(shape=(n_data, n_states))
cn = np.empty(n_data)
alpha_base = np.empty(n_states)
beta_base = np.empty(n_states)
delta = np.empty(shape=(n_data, n_states))
epsilon = np.empty(shape=(n_data, n_states))

iter_total = 10

iter_no = 1

while iter_no <= iter_total:

#********************************EXPECTATION STEP*****************************#

    # Calculating alpha_base, cn[0] and base for alpha_sc
    alpha_base = A[0,:]
    cn[0] = np.sum(pi*B[0,:])
    alpha_sc[0,:] = alpha_base/cn[0]
    delta[0,:] = alpha_sc[0,:]*cn[0]
    
    for i in range(1, n_data):
        for j in range(n_states):
            loc = np.where(x[i] == x_states)[0]
            emis_prob = B[loc, j]
            delta[i,j] = emis_prob*np.sum(alpha_sc[i-1,:]*A[:,j])
        cn[i] = np.sum(delta[i,:])
        alpha_sc[i,:] = delta[i,:]/cn[i]
    
    beta_base = 1
    beta_sc[-1,:] = beta_base
    
    # epsilon[-1,:] not initialized with values. Thus do not use it anywhere !
    for i in range(n_data-2, -1, -1):
        for j in range(n_states):
            loc = np.where(x[i+1] == x_states)[0]
            emis_prob = B[loc, j]
            epsilon[i,j] = np.sum(beta_sc[i+1,:]*emis_prob*A[j,:])
        beta_sc[i,:] = epsilon[i,:]/cn[i+1]
    
#********************************MAXIMIZATION STEP****************************#
    
    # Initial probability vector pi
    pi = alpha_sc[0,:]*beta_sc[0,:]/np.sum(alpha_sc[0,:]*beta_sc[0,:])
    
    # Transition probability matrix A
    Atemp = np.empty_like(A)
    for i in range(n_states):
        for j in range(n_states):
            numerator = 0
            denominator = 0
            for n in range(0, n_data-1):
                loc = np.where(x[n] == x_states)[0]
                emis_prob = B[loc,i]
                numerator = numerator + alpha_sc[n-1,i]*beta_sc[n,j]*emis_prob*A[i,j]/cn[n]
            for l in range(n_states): 
                temp = 0
                for n in range(0, n_data-1):
                    loc = np.where(x[n] == x_states)[0]
                    emis_prob = B[loc,l]
                    temp = temp + alpha_sc[n-1,i]*beta_sc[n,l]*emis_prob*A[i,l]/cn[n]
                denominator = denominator + temp
            Atemp[i,j] = numerator/denominator
    
    A[:] = Atemp
    
    # Emission probability matrix B
    for i in range(n_xstates):
        for j in range(n_states):
            arg_list = np.where(x == x_states[i])[0]
            #numerator = np.sum([alpha_sc[arg_list[e],j]*beta_sc[arg_list[e],j]*x[arg_list[e]] for e in range(len(arg_list))])
            numerator = np.sum([alpha_sc[e,j]*beta_sc[e,j]*x[e] for e in arg_list])
            B[i,j] = numerator/np.sum(alpha_sc[:,j]*beta_sc[:,j])

    print(iter_no)
    iter_no = iter_no + 1