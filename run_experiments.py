import numpy as np
import os, itertools
import seaborn as sns
from src import fileio
import matplotlib.pyplot as plt
from collections import defaultdict

param_combos = list(itertools.product(
    ['n_step_expected_sarsa', 'tree_backup', 'qsigma'], # agents
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], #np.linspace(0.1, 0.9, num=9), # learning rates
    [0.1], #, 0.2, 0.3], # epsilons
    [10], #[5, 10] # nsteps
    [0.25, 0.5, 0.75, 2.0] #, 3.0] # sigmas
))

results_learning = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))
results_learning_sigmas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: (0, 0, 0))))
results_post = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))
for i, combo in enumerate(param_combos):
    print("Combo ", i, ":", combo)
    agent, alpha, epsilon, n_step, sigma = combo
    
    if agent is not 'qsigma':
        sigma = 0.0
    
    #result_learning_fname = "results/returns_learning_" + agent + "_" + str(alpha) + "_" + str(epsilon) + "_" + str(n_step) + "_" + str(sigma) + ".txt"
    
    result_post_fname = "results/returns_post_" + agent + "_" + str(alpha) + "_" + str(epsilon) + "_" + str(n_step) + "_" + str(sigma) + ".txt"
    
    # train the model and then, after finished learning of 100 eps, run 1 episode and save return to file
    if not os.path.exists(result_post_fname):
        for i in range(100):
            os.system("python3 src/gridworld.py -a " + agent + " -n 0 -k 100 -Z 1 -g WindyGrid -B " + str(n_step) + " -w 90 -s 100 -e " + str(epsilon) + " -l " + str(alpha) + " -O " + str(sigma))
    
    '''
    print(result_learning_fname)
    assert os.path.exists(result_learning_fname)
       
    tuples = [(int(str(x.split(",")[0][1:])), float(str(x.split(", ")[1][0:-2]))) for x in fileio.read_line_list(result_learning_fname, load_float=False)]
    timesteps, returns = [x[0] for x in tuples], [x[1] for x in tuples]
    avg_return = sum(returns[:50])/50 # np.average(returns[:50])
    
    if avg_return > results_learning[agent][alpha][2]: # if this eps gives a better avg. return, then plot that instead.
        results_learning[agent][alpha] = (epsilon, sigma, avg_return)
    
    if agent is 'qsigma':
        if avg_return > results_learning_sigmas[agent][sigma][alpha][2]:
            results_learning_sigmas[agent][sigma][alpha] = (epsilon, avg_return)
    '''
    
    assert os.path.exists(result_post_fname)
    
    tuples = [(int(str(x.split(",")[0][1:])), float(str(x.split(", ")[1][0:-2]))) for x in fileio.read_line_list(result_post_fname, load_float=False)]
    timesteps, returns = [x[0] for x in tuples], [x[1] for x in tuples]
    avg_return = sum(returns)/len(returns) # np.average(returns)
    
    if avg_return > results_post[agent][alpha][2]: # if this eps gives a better avg. return, then plot that instead.
        results_post[agent][alpha] = (epsilon, sigma, avg_return)
    
    if agent is 'qsigma':
        if avg_return > results_learning_sigmas[agent][sigma][alpha][2]:
            results_learning_sigmas[agent][sigma][alpha] = (epsilon, avg_return)

plt.figure()
plt.title('Avg. return per alpha value')
plt.ylabel('Avg. total return over 120 runs after 100 episodes')
plt.xlabel('Learning rate')
plt.xlim(0, 1)
plt.ylim(0, 0.4)

for agent in results_post:
    alphas, avg_returns = [], []
    for alpha in results_post[agent]:
        alphas.append(alpha)
        epsilon, sigma, avg_return = results_post[agent][alpha]
        avg_returns.append(avg_return)
        print(agent, alpha, epsilon, sigma)
    
    #agent_results = results_post[agent]
    #alphas, avg_returns = [x[0] for x in agent_results], [x[1] for x in agent_results]
    plt.plot(alphas, avg_returns, '*-', label=agent)
    
plt.legend()
plt.savefig('returns_per_alpha.png', bbox_inches='tight')

# graph sigma vals.

plt.figure()
plt.title('Avg. return using Q(sigma) per sigma & alpha')
plt.ylabel('Avg. total return over 120 runs after 100 episodes')
plt.xlabel('Learning rate')
plt.xlim(0, 1)
plt.ylim(0, 0.4)

for sigma in results_learning_sigmas['qsigma']:
    alphas, avg_returns = [], []
    for alpha in results_learning_sigmas['qsigma'][sigma]:
        alphas.append(alpha)
        epsilon, avg_return = results_learning_sigmas['qsigma'][sigma][alpha]
        avg_returns.append(avg_return)
        print(agent, alpha, epsilon, sigma)
    
    #agent_results = results_post[agent]
    #alphas, avg_returns = [x[0] for x in agent_results], [x[1] for x in agent_results]
    label = sigma
    if sigma == 2: label = 'linear decay'
    elif sigma == 3: label = 'exp. decay'
    plt.plot(alphas, avg_returns, '*-', label=label)
    
plt.legend()
plt.savefig('returns_per_sigma.png', bbox_inches='tight')