#calculate probability based on pdf and then draw pictures
from scipy.stats import norm
from scipy.integrate import trapz
import numpy as np
import matplotlib.pyplot as plt
'''
x1 = np.linspace(-2,2,100)
p = trapz(norm.pdf(x1),x1)
print '{:.2%}'.format(p)
fb = plt.fill_between(x1,norm.pdf(x1),color = 'gray')
x=np.linspace(-3,3,50)
p = plt.plot(x,norm.pdf(x),'k-')
plt.show()

x = np.linspace(-10,10,100)
P = plt.plot(x,norm.pdf(x,loc = 0.5,scale = 2))
P = plt.plot(x,norm.pdf(x,loc = 3,scale = 2))
x1 = np.linspace(-3.5,4.5,20)
plt.fill_between(x1,norm.pdf(x1,0.5,2),color = 'gray')
plt.show()
'''
x = range(10)
y = [3,4,2,3,4,5,4,3,4,5]
plt.stem(x,y)
plt.show()