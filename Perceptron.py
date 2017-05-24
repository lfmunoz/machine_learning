#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from random import choice

import logging
logging.basicConfig(level=logging.DEBUG) 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.info('**** Starting Application ****')

def gen_y_vec(w,x):
    return [((-w[0][1]*num) - w[0][0]) / float(w[0][2]) for num in x]

def check_classification(w,x,y):
    if np.sign(np.dot(w, x.T)) != y:
        logging.debug('classification is NOT correct')
        return False
    logging.debug('classification is correct')
    return True


def check_trainingSet(w, x ,y, debug=False):
    logging.info('** Checking entire set **')
    for i in range(len(x)):
        if np.sign(np.dot(w, x[i].T)) != y[i]:
            logging.debug('missclassified x = %s i = %s', x[i], i)
            if (not debug): return False
    if (not debug): logging.debug('all training examples correctly classified!!!')
    if (not debug): return True

def modify_weight(w,x,y):
    logging.info('* Modifying Weight *')
    logging.debug('starting w = %s, x = %s, y = %s ', w, x, y)
    for idx in range(len(w[0])):
        w[0][idx] = w[0][idx] + .2*x[0][idx]*y





def perceptron_algo(w,x,y):
    logging.info('*** Running Perceptron Learning Algorithm ***')
    logging.debug('w = %s shape = %s', w, w.shape);
    logging.debug('x = \n %s', x);
    logging.debug('x element shape = %s', x[0].shape);
    logging.debug('y = %s', y);

    check = True 

#    clist = [0,5,5,4,3,2,2,1,1]
#    i = 0
    while(check) :
        randIndx  = choice(range(len(x)))
        #randIndx  = clist[i]
        #i = i + 1
        sample = x[randIndx]
        target = y[randIndx]
        logging.debug('randIndx = %s, w = %s, x = %s, y = %s', randIndx, w, sample, target);
        # if classification is correct, check entire set
        if check_classification(w, sample, target):
            check = not check_trainingSet(w, x,y)
        # if classification is not correct, modify weight
        else:
            modify_weight(w,sample,target)
            logging.debug('modified w = %s', w)



# dimension (not used)
d = 2
# initial 'random' solution
w = np.array([[0, 1 , .5]])
# the training set
x = [
np.array([[1,1,1]]),
np.array([[1,-2,1]]),
np.array([[1,1.5,-1]]),
np.array([[1,-2,-1]]),
np.array([[1,-1,-1.5]]),
np.array([[1,2,-2]])
]
y = np.array([1,1,1, -1,-1,-1])

# 0, 6, 6, 5, 4,3,3,2,2

#print "Testing"
#print w
#modify_weight(w,x[0],y[4])
#print w

circle_x = [x[idx][0][1] for idx in range(len(x)) if y[idx] == 1]
circle_y = [x[idx][0][2] for idx in range(len(x)) if y[idx] == 1]
square_x = [x[idx][0][1] for idx in range(len(x)) if y[idx] == -1]
square_y = [x[idx][0][2] for idx in range(len(x)) if y[idx] == -1]

#print np.dot(xvec[1], w.T)

lineXvec = np.arange(-3,3,0.1)
lineYvec = gen_y_vec(w,lineXvec)

 
perceptron_algo(w,x,y)

lineXsol = np.arange(-3,3,0.1)
lineYsol = gen_y_vec(w,lineXsol)


fig1 = plt.figure(figsize=(10,8), dpi=100)
#ax = fig.gca()
#ax.set_xticks(np.arange(0,1,0.1))

#Create a new subplot from a grid of 1x1
plt.subplot(111)
#Plot the y_n = 1 points
plt.scatter(circle_x,circle_y, marker='o', color='r')
#Plot the y_n = -1 points
plt.scatter(square_x,square_y, marker='s')
# plot the h line
plt.plot(lineXvec,lineYvec)

plt.plot(lineXsol,lineYsol)

ax = plt.gca()
# set the x-spine 
ax.spines['left'].set_position('zero')
# turn off the right spine/ticks
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()
# set the y-spine
ax.spines['bottom'].set_position('zero')
# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()
ax.set_aspect('equal')
plt.grid()
plt.show()
#fig1.show()
#raw_input()

#check_trainingSet(w,x,y, debug=True)
#wt = [[1,2,3]]
#xt = [[1,2,3]]
#modify_weight(wt, xt, 2)
#print wt



print "Press any key to end application"
#https://www.labri.fr/perso/nrougier/teaching/matplotlib/
