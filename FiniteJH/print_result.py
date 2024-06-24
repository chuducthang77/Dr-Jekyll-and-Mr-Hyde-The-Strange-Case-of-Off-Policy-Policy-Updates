import numpy as np
import pprint
import matplotlib.pyplot as plt
import os
import pickle

# Input: fig-2a pickle
# Output: fig-2a graph
existing = True

if existing:
    # Load the existing pickle file
    result = np.load('./expes/fig-2a/curves/stats_of_interest.pkl', allow_pickle=True)

    # Print all the current keys in the dictionary. These are all compared methods (PG, J&H) with different hyperparameters and variations
    print('Result keys')
    pprint.pprint(result.keys())
    print('---------------------------')
    # Print the result of a given method
    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$'])
    print('---------------------------')

    # Print the keys of a given method's dictionary. There are two keys 'perf_glob' and 'perf_jekyll'. In the 'sample' setting (does not know the true r and P), 'perf_glob' refers to 'theory' and 'practice' (basically 0 discounting) discounting, while 'perf_jekyll' refers to 'jh' (ot in the paper) discounting. In the 'true' setting, they are all 'perf_glob'.
    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - keys')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$'].keys())
    print('---------------------------')

    # Print the results of a specific key. Each key has a dictionary value, which contains another 4 keys 'mean', 'median', 'decile', 'percentile'
    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_glob')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob'])
    print('---------------------------')

    # Print the results of a specific key out of 4 above keys and the length of its array value. The length of the array is 1001.
    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_glob -  median')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['median'])
    print(len(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['median']))
    print('---------------------------')

    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_glob - mean')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['mean'])
    print(len(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['mean']))
    print('---------------------------')

    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_glob -  decile')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['decile'])
    print(len(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['decile']))
    print('---------------------------')

    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_glob - percentile')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['percentile'])
    print(len(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['percentile']))
    print('---------------------------')

    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_jekyll')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_jekyll'])
    print('---------------------------')

    print('J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$ - perf_jekyll -  median')
    print(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_jekyll']['median'])
    print(len(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_jekyll']['median']))
    print('---------------------------')

    # plt.plot(result['J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$']['perf_glob']['mean'],
    #          label='J\\&H $\\epsilon_t=100/t$, $o_{t}=0.5$')
    # plt.plot(result['J\\&H $\\epsilon_t=10/\\sqrt{t}$, $o_{t}=0.5$']['perf_glob']['mean'],
    #          label='J\\&H $\\epsilon_t=10/\\sqrt{t}$, $o_{t}=0.5$')
    # plt.plot(result['J\\&H $\\epsilon_t=10/\\sqrt{t}$, $o_{t}=10/\\sqrt{t}$']['perf_glob']['mean'],
    #          label='J\\&H $\\epsilon_t=10/\\sqrt{t}$, $o_{t}=10/\\sqrt{t}$')
    # plt.plot(result['PG update $\\lambda=0$']['perf_glob']['mean'], label='PG update $\\lambda=0$')
    # plt.plot(result['PG update $\\lambda=0.01$']['perf_glob']['mean'], label='PG update $\\lambda=0.01$')
    # plt.plot(result['undiscounted update $\\lambda=0$']['perf_glob']['mean'], label='undiscounted update $\\lambda=0$')
    plt.plot(result['PG update $\\nu=1$']['perf_glob']['mean'], label='PG update $\\nu=1$')
    plt.plot(result['PG update $\\nu=0.1$']['perf_glob']['mean'], label='PG update $\\nu=0.1$')
    plt.plot(result['PG update $\\nu=0.01$']['perf_glob']['mean'], label='PG update $\\nu=0.01$')
    plt.legend()
    plt.show()
else:
    # Load a single result to understand the structure of output
    # result = np.load('./expes/fig-2a/config_chain_sample_64162002.pkl', allow_pickle=True)        # # Print the keys of the training pickle. There are 5 keys cfg, seed, env, p_star, p_rand
    # print(result.keys())
    # print('---------------------------')
    # # Print the keys of result cfg. There are many keys (similar to config file)
    # print(result['cfg'].keys())
    # print('---------------------------')
    # # Print the cfg algo result. The printing result is a list of 3 dictionaries. There are 3 dictionaries to compare 3 different critic_UCB_stepsize The keys of these 3 dictionaries are similar.
    # print(result['cfg']['algos'][0].keys())
    # # Print the result of training algo. However, there are no mean, median, decile, and percentile as above
    # print('---------------------------')
    # print(result['cfg']['algos'][0]['res'].keys())
    # print(len(result['cfg']['algos'][0]['res']['perf_glob']))
    # # print(result['cfg']['algos'][0]['res']['perf_glob'])
    # print('---------------------------')
    # Load the training file    # Load all 100 results to calculate the mean
    dir = './expes/fig-2a/'
    result = {}
    for name in os.listdir(dir):
        if '.pkl' in name:
            res = np.load(dir + str(name), allow_pickle=True)
            for i in range(len(res['cfg']['algos'])):
                if res['cfg']['algos'][i]['name'] not in result:
                    result[res['cfg']['algos'][i]['name']] = [res['cfg']['algos'][i]['res']['perf_glob']]
                else:
                    result[res['cfg']['algos'][i]['name']].append(res['cfg']['algos'][i]['res']['perf_glob'])

                # Calculate the mean of 100 results
    for key in result.keys():
        result[key] = np.array(result[key])
        result[key] = np.mean(result[key], axis=0)
    print(result)
    # Plot the result
    for key in result.keys():
        plt.plot(result[key], label=key)
    plt.legend()
    plt.show()
    # Save the result to transfer among machines
    # with open('result.pkl', 'wb') as fp:
    #  pickle.dump(result, fp)
