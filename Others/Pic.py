# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:54:00 2016

@author: ZJun
"""


import matplotlib.pyplot as plt


Image = plt.imread('IMG_1559 2.JPG')

plt.imshow(Image)


I1 = Image[:,:,0]
I2 = Image[:,:,1]
I3 = Image[:,:,2]



New_Pic = np.dstack([I1,I2,I3])

plt.imshow(New_Pic)


def Change_Matrix_value(Matrix , Big_Than , Change_to):
    x,y = Matrix.shape
    for i in range(x):
        for j in range(y/2,y):
            if Matrix[i,j] > Big_Than:
                Matrix[i,j] = Change_to
    return Matrix
    
def Change_Matrix_value(Matrix , Big_Than , Change_to):
    x,y = Matrix.shape
    for i in range(x):
        for j in range(y):
            Matrix[i,j] = Matrix[x-1-i,j]
    return Matrix

    
Change_Matrix_value(I1,100,100)
Change_Matrix_value(I2,100,100)
Change_Matrix_value(I3,100,100)

plt.imshow(Image)
