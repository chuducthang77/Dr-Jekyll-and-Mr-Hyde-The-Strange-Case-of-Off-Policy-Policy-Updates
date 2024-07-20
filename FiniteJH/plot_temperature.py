import numpy as np
import pprint
import matplotlib.pyplot as plt
import os
import pickle

dir = './expes/fig_new_exact/13_exp_plot_temperature_average_runs/'
result = {}
for name in os.listdir(dir):
    if '.pkl' in name:
        res = np.load(dir + str(name), allow_pickle=True)
        for i in range(len(res['cfg']['algos'])):
            if res['cfg']['algos'][i]['name'] not in result:
                result[res['cfg']['algos'][i]['name']] = [res['cfg']['algos'][i]['res']['temperature']]
            else:
                result[res['cfg']['algos'][i]['name']].append(res['cfg']['algos'][i]['res']['temperature'])

# Calculate the mean of 100 results
nb_state = 0
for key in result.keys():
    result[key] = np.array(result[key])
    result[key] = np.mean(result[key], axis=0)
    nb_state = result[key].shape[1]
    print(result[key].shape)


for i in range(nb_state):
    plt.figure()
    for key in result.keys():
        plt.plot(result[key][:, i], label = key)
        print('State {i} algo {key} temp {temp}'.format(i=i,key=key, temp=result[key][:, i]))
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Temperature in log scale')
    plt.xlabel('Time step')
    plt.title('Temperature for state {i}'.format(i=(i+1)))
    plt.savefig('expes/fig_new_exact/13_exp_plot_temperature_average_runs/Temperature for state {i}.png'.format(i=(i+1)))

# print(result)