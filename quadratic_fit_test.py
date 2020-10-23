import numpy as np
import matplotlib.pyplot as plt

## Sample within unit circle R^2
length = np.sqrt(np.random.uniform(0,1,400))
print(length)

theta = np.pi * np.random.uniform(0,2,400)
print(theta)

x=length * np.cos(theta)
y=length * np.sin(theta)
z=np.sqrt(1-x**2-y**2)

plt.scatter(x,y)
plt.show()

table=np.stack((x,y,z),axis=1)

print(table)

