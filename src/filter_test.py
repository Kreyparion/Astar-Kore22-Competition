import imp
from re import L
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from math import exp,log

a = [[[0 for i in range(20)] for j in range(20)] for loop in range(40)]
for i in range(len(a)):
    pass
    #a[i][5][5] = 100
"""
    a[i][6][6] = 100
    a[i][5][6] = 30
    a[i][11][11] = 200
    a[i][11][12] = 200
    a[i][11][13] = 200
    a[i][12][13] = 200
    a[i][13][13] = 200
    
    a[i][10][13] = 200
    a[i][9][13] = 200
    a[i][11][14] = 200
    a[i][11][15] = 200
"""

# a = [[[0, 0, 20, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 20, 0, 0], [0, 0, 0, 0, 0, 0, 11, 54, 0, 0, 0, 0, 0, 54, 11, 0, 0, 0, 0, 0, 0], [20, 0, 14, 0, 0, 16, 0, 19, 0, 0, 0, 0, 0, 19, 0, 16, 0, 0, 14, 0, 20], [0, 0, 0, 16, 0, 0, 1, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 16, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 16, 0, 0, 0, 0, 1, 38, 0, 0, 0, 38, 1, 0, 0, 0, 0, 16, 0, 0], [0, 11, 0, 1, 0, 10, 0, 15, 5, 0, 0, 0, 5, 15, 0, 10, 0, 1, 0, 11, 0], [0, 54, 19, 0, 16, 0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 16, 0, 19, 54, 0], [0, 0, 0, 0, 66, 6, 5, 8, 10, 0, 0, 0, 10, 8, 5, 6, 66, 0, 0, 0, 0], [3, 0, 0, 30, 11, 2, 0, 0, 0, 13, 51, 13, 0, 0, 0, 2, 11, 30, 0, 0, 3], [0, 0, 0, 3, 21, 17, 5, 14, 0, 4, 0, 4, 0, 14, 5, 17, 21, 3, 0, 0, 0], [3, 0, 0, 30, 11, 2, 0, 0, 0, 13, 51, 13, 0, 0, 0, 2, 11, 30, 0, 0, 3], [0, 0, 0, 0, 66, 6, 5, 8, 10, 0, 0, 0, 10, 8, 5, 6, 66, 0, 0, 0, 0], [0, 54, 19, 0, 16, 0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 16, 0, 19, 54, 0], [0, 11, 0, 1, 0, 10, 0, 15, 5, 0, 0, 0, 5, 15, 0, 10, 0, 1, 0, 11, 0], [0, 0, 16, 0, 0, 0, 0, 1, 38, 0, 0, 0, 38, 1, 0, 0, 0, 0, 16, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 16, 0, 0, 1, 0, 0, 0, 19, 0, 0, 0, 1, 0, 0, 16, 0, 0, 0], [20, 0, 14, 0, 0, 16, 0, 19, 0, 0, 0, 0, 0, 19, 0, 16, 0, 0, 14, 0, 20], [0, 0, 0, 0, 0, 0, 11, 54, 0, 0, 0, 0, 0, 54, 11, 0, 0, 0, 0, 0, 0], [0, 0, 20, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 20, 0, 0]] for _ in range(40)]
# a = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0], [0, 10, 0, 0, 0, 0, 27, 9, 28, 0, 0, 0, 28, 9, 27, 0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 34, 0, 0, 2, 0, 7, 0, 7, 0, 2, 0, 0, 34, 0, 0, 0, 0], [0, 0, 0, 34, 0, 0, 0, 25, 18, 10, 12, 10, 18, 25, 0, 0, 0, 34, 0, 0, 0], [0, 0, 0, 0, 0, 1, 41, 0, 12, 0, 0, 0, 12, 0, 41, 1, 0, 0, 0, 0, 0], [0, 0, 27, 0, 0, 0, 0, 7, 2, 5, 0, 5, 2, 7, 0, 0, 0, 0, 27, 0, 0], [0, 0, 9, 2, 17, 5, 0, 21, 4, 3, 6, 3, 4, 21, 0, 5, 17, 2, 9, 0, 0], [0, 0, 28, 6, 0, 2, 0, 8, 0, 5, 0, 5, 0, 8, 0, 2, 0, 6, 28, 0, 0], [0, 0, 0, 0, 28, 0, 0, 0, 29, 8, 13, 8, 29, 0, 0, 0, 28, 0, 0, 0, 0], [0, 6, 0, 3, 0, 17, 0, 0, 36, 370, 0, 370, 36, 0, 0, 17, 0, 3, 0, 6, 0], [0, 0, 0, 0, 28, 0, 0, 0, 29, 8, 13, 8, 29, 0, 0, 0, 28, 0, 0, 0, 0], [0, 0, 28, 6, 0, 2, 0, 8, 0, 5, 0, 5, 0, 8, 0, 2, 0, 6, 28, 0, 0], [0, 0, 9, 2, 17, 5, 0, 21, 4, 3, 6, 3, 4, 21, 0, 5, 17, 2, 9, 0, 0], [0, 0, 27, 0, 0, 0, 0, 7, 2, 5, 0, 5, 2, 7, 0, 0, 0, 0, 27, 0, 0], [0, 0, 0, 0, 0, 1, 41, 0, 12, 0, 0, 0, 12, 0, 41, 1, 0, 0, 0, 0, 0], [0, 0, 0, 34, 0, 0, 0, 25, 18, 10, 12, 10, 18, 25, 0, 0, 0, 34, 0, 0, 0], [0, 0, 0, 0, 34, 0, 0, 2, 0, 7, 0, 7, 0, 2, 0, 0, 34, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0, 27, 9, 28, 0, 0, 0, 28, 9, 27, 0, 0, 0, 0, 10, 0], [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] for _ in range(40)]
permanent_a = [[] for _ in range(len(a))]
for i in range(len(a)):
    permanent_a[i] = deepcopy(a[0])
    permanent_a[i][5][(5+i)%20] = 100

filter_perfect = [[0,0,1.3,0,0],
          [0,0,1.8,0,0],
          [1.3,1.8,-3.22,1.8,1.3],
          [0,0,1.8,0,0],
          [0,0,1.3,0,0]]

filter_static_perfect = [[-0.3,-0.24,1.7,-0.24,-0.3],
          [-0.24,-0.4,1.9,-0.4,-0.2],
          [1.7,1.9,-0.75,1.9,1.7],
          [-0.24,-0.4,1.9,-0.4,-0.24],
          [-0.3,-0.24,1.7,-0.24,-0.3]]



# Really good filter but a little bit too noisy

filter_2_perfect = [[-0.37,-0.32,1.5,-0.32,-0.37],
          [-0.32,-0.45,2.1,-0.45,-0.32],
          [1.5,2.1,-0.75,2.1,1.5],
          [-0.32,-0.45,2.1,-0.45,-0.32],
          [-0.37,-0.3,1.5,-0.32,-0.37]]

"""
cc = -0.05
cd = -0.4
ld = -0.5
c = -0.3
d = 2.25
dd = 1.3
filter = [[cc,ld,dd,ld,cc],
          [ld,cd,d,cd,ld],
          [dd,d,c,d,dd],
          [ld,cd,d,cd,ld],
          [cc,ld,dd,ld,cc]]
"""
"""
# Filter trying to suppress surrounding noise -> negative values in the corners
dimin = 0.85
cc = -0.2 * dimin
cd = -0.6 * dimin
ld = -0.6 * dimin
c = -0.8 * dimin
d = 2.4 * dimin
dd = 1.6 * dimin
ratio = 7.3/10
filter = [[cc,ld,dd,ld,cc],
          [ld,cd,d,cd,ld],
          [dd,d,c,d,dd],
          [ld,cd,d,cd,ld],
          [cc,ld,dd,ld,cc]]

best filter but decreases too fast
"""

"""
/!\filtre violent
dimin = 1
cc = -0.4 * dimin
cd = -1.15 * dimin
ld = -0.6 * dimin
c = -0.8 * dimin
d = 3.3 * dimin
dd = 1.8 * dimin
ratio = 1.7/10

filter = [[cc,ld,dd,ld,cc],
          [ld,cd,d,cd,ld],
          [dd,d,c,d,dd],
          [ld,cd,d,cd,ld],
          [cc,ld,dd,ld,cc]]

"""

# better filter : balanced + long range (not too powerful in long range)
dimin = 1
cc = -0.2 * dimin
cd = -1 * dimin
ld = -0.5 * dimin
c = -0.8 * dimin
d = 3 * dimin
dd = 1.4 * dimin
ratio = 4.5/10

filter = [[cc,ld,dd,ld,cc],
          [ld,cd,d,cd,ld],
          [dd,d,c,d,dd],
          [ld,cd,d,cd,ld],
          [cc,ld,dd,ld,cc]]

e = 0.6
i = 1.6
filter_small2 = [[0,0,e,0,0],
          [0,0,i,0,0],
          [e,i,0,i,e],
          [0,0,i,0,0],
          [0,0,e,0,0]]
f = 2.2
filter_small = [[0,f,0],
          [f,0,f],
          [0,f,0]]

e = 1.8
i = 1.6
c = -2
filter_hmm = [[0,0,e,0,0],
          [0,0,i,0,0],
          [e,i,c,i,e],
          [0,0,i,0,0],
          [0,0,e,0,0]]

def use_filter(a,n,filter):
    for i in range(len(a[0])):
        for j in range(len(a[0][0])):
            sum = 0
            for k in range(len(filter)):
                for l in range(len(filter[0])):
                    t = n+abs(-k+len(filter)//2)+abs(-l+len(filter)//2)
                    if t <0:
                        t = 0
                    x = (i-k+len(filter)//2)%len(a[0])
                    y = (j-l+len(filter[0])//2)%len(a[0][0])
                    sum += a[t][x][y]*filter[k][l]
            a[n][i][j] = (sum + permanent_a[i][j]*7.24)/10

def use_t_filter(a,n,filter):
    for i in range(len(a[0])):
        for j in range(len(a[0][0])):
            sum = 0
            for k in range(len(filter)):
                for l in range(len(filter[0])):
                    t = n+abs(-k+len(filter)//2)+abs(-l+len(filter)//2)
                    if t>len(a)-1:
                        t = len(a)-1
                    x = (i-k+len(filter)//2)%len(a[0])
                    y = (j-l+len(filter[0])//2)%len(a[0][0])
                    sum += a[t][x][y]*filter[k][l]
            a[n][i][j] = (sum + permanent_a[i][j]*7.5)/10

def use_t_filter2(a,n,filter):
    for i in range(len(a[0])):
        for j in range(len(a[0][0])):
            sum = 0
            for k in range(len(filter)):
                for l in range(len(filter[0])):
                    t = n+abs(-k+len(filter)//2)+abs(-l+len(filter)//2)
                    if t>len(a)-1:
                        t = len(a)-1
                    x = (i-k+len(filter)//2)%len(a[0])
                    y = (j-l+len(filter[0])//2)%len(a[0][0])
                    sum += a[t][x][y]*filter[k][l]
            a[n][i][j] += sum/10
"""
for k in range(len(permanent_a)):
    for l in range(len(permanent_a[0])):
        permanent_a[k][l] *= 1.02**(len(a))
"""

for n in range(len(a)-1,-1,-1):
    for i in range(len(a[0])):
        for j in range(len(a[0][0])):
            a[n][i][j] = permanent_a[n][i][j]*ratio

for i in range(len(a)-1,-1,-1):
    use_t_filter2(a,i,filter)
    """
    for k in range(len(permanent_a)):
        for l in range(len(permanent_a[0])):
            permanent_a[k][l] *= 0.98
    """

plt.matshow(a[0])
plt.matshow(a[1])
plt.show()

"""
X = list(range(len(a)))
Y = [a[i][5][5] for i in X]
import scipy.interpolate
y_interp = scipy.interpolate.interp1d(X,Y)
Y_real = [100/Y[i] for i in X]
# myY = [exp(i/6.2)/200+1 for i in X]
r = ratio
k = 30
v = k*(len(a)*r*(len(a)-1))/(1-len(a)*r)
u = -v/(len(a))
j = v
c = 0.0015

myY = [(k*i+j)/(u*i+v)+c*i for i in X]
print(myY[-2])
#plt.plot(X,Y,"b")
plt.plot(X,Y_real,"r")
plt.plot(X,myY,"y")
plt.show()
"""
"""
plt.matshow(a[30])
plt.show()
plt.matshow(permanent_a)
plt.show()
"""