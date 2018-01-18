import numpy as np
import matplotlib.pyplot as plt

loss= [0.1104, 
       0.0894,
       0.0780, 
       0.0673,
       0.0581,
       0.0503,
       0.0452,
       0.0385,
       0.0347,
       0.0302,
       0.0262,
       0.0229,
       0.0211,
       0.0183,
       0.0174,
       0.0153,
       0.0143,
       0.0163,
       0.0128,
       0.0111,
       0.0105,
       0.0122,
       0.0115,
       0.0095,
       0.0086]
       
       
       
       
       
valLoss= [ 0.1040, 
           0.0901,
           0.0774,
           0.0675,
           0.0649,
           0.0587,
           0.0555,
           0.0546,
           0.0544,
           0.0517,
           0.0501,
           0.0479,
           0.0505,
           0.0506,
           0.0483,
           0.0478,
           0.0444,
           0.0465,
           0.0456,
           0.0453,
           0.0450,
           0.0507,
           0.0470,
           0.0464,
           0.0444]

### plot the training and validation loss for each epoch
plt.plot(loss)
plt.plot(valLoss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()