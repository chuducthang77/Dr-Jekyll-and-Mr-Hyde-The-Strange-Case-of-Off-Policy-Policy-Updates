import numpy as np
import matplotlib.pyplot as plt

# Understand the result of pickle
result = np.load('./results/Four_Rooms_Static.pkl', allow_pickle=True)

# Print the keys of the result. Only the name of the algorithm 'JH'
# print(result.keys())

# Print the result of result['JH']. This is a list of only 1 item, containing 5 small items
print(len(result['JH']))

# Print the result of result['JH'][0]. This is a list of 5 small items. The first
# 2 items are a list of 491 items, the last 3 items are single integer. 
print(len(result['JH'][0]))