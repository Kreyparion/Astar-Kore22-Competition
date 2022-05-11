import matplotlib.pyplot as plt
import numpy as np

a = [[[0 for i in range(20)] for j in range(20)] for loop in range(40)]
for i in range(len(a)):
    a[i][5][5] = 100
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
permanent_a = a[0]

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




filter = [[-0.37,-0.32,1.5,-0.32,-0.37],
          [-0.32,-0.45,2.1,-0.45,-0.32],
          [1.5,2.1,-0.75,2.1,1.5],
          [-0.32,-0.45,2.1,-0.45,-0.32],
          [-0.37,-0.3,1.5,-0.32,-0.37]]


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
            a[n][i][j] = (sum + permanent_a[i][j]*7.2)/10


for i in range(39,-1,-1):
    use_t_filter(a,i,filter)

plt.matshow(a[0])
plt.show()
plt.matshow(a[30])
plt.show()