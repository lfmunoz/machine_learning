#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl


x = np.array()


fig = plt.figure(figsize=(8,6), dpi=80)

plt.subplot(111)

plt.scatter(x_0,y_0, marker='o', color='r')

plt.scatter(x_1,y_1, marker='s')

ax = plt.gca()
plt.grid()
plt.show()


