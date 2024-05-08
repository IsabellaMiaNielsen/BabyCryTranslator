import pickle
import numpy as np
import matplotlib.pyplot as plt

# All the features are 2D arrays
features = pickle.load(open('extracted_features.pkl', 'rb'))

# Plot features
first =  True
fig, ax = plt.subplots(nrows=7)
for key, color in zip(features, ["b", "g", "r", "c", "m", "y", "k", "grey", "brown"]): # Label
    first_key = True
    for key2, round in zip(features[key], range(len(features[key]))): # File name
        for key3, index in zip(features[key][key2], [0, 1, 2, 3, 4, 5, 6, 7, 8]): # Feature name
            if index < 2:
                continue
            data = np.array(features[key][key2][key3])
            if first:
                ax[index - 2].set(title=key3)
                print("key: ", key3, " dimensions: ", data.ndim)
            if first_key:
                
                ax[index - 2].scatter(data[:, 0], data[:, 1], label=key, color=color)
                first_key = False
                print("First key for ", key)
            else:
                ax[index - 2].scatter(data[:, 0], data[:, 1], color=color)
        first = False
        break
plt.show()

