## Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

## Create PDF Functions


#Uniform PDF

def uniform(n,d):
    '''
    d (int) - dimensions
    n (int) - number of points
    '''

    sample=np.random.rand(n,d) #Obtain sample

    if (n==1): #Output only the point (not list) if only one point is desired
        return(sample[0])
    else:   #Output list of points if multiple are desired
        return(sample)


def gaussian(n,d):
    '''
    d (int) - dimensions
    n (int) - number of points
    '''

    sample=np.random.randn(n,d)

    if (n==1):
        return(sample[0])
    else:
        return(sample)


def x_5(n,d=1):
    if (n==1):
        sample=np.random.rand(n,d)[0] 
    else:
        sample=np.random.rand(n,d)
   
    def inv_CDF(x):
        return(x**(1/6))

    return(inv_CDF(sample))

def x_1(n,d=1):
    if (n==1):
        sample=np.random.rand(n,d)[0]
    else:
        sample=np.random.rand(n,d)

    def inv_CDF(x):
        return(np.sqrt(x))

    return(inv_CDF(sample))

x5=x_5(1000)

#print(x5)

plt.hist(x5)
plt.show()

#max=np.amax(x1)
#plt.hist(x1/max)
#plt.show()
