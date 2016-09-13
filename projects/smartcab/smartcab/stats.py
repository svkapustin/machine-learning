import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import sys
from cStringIO import StringIO

bad = re.compile('ran out')
good = re.compile('reached')
stats = re.compile('(STAT.*)')

bad_c = 0
good_c = 0
stats_list = []
stats_str = StringIO()

for line in sys.stdin:
    if bad.search(line):
        bad_c += 1
    if good.search(line):
        good_c += 1
    if stats.search(line):
        stats_str.write(line)

print 'Reached: {}, timed out: {}'.format(good_c, bad_c)

stats_str.seek(0)
d = pd.read_csv(stats_str)
f, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,12))

cols = ['Rewards', 'Steps']
for i in range(len(cols)):
    med = d[cols[i]].describe()['50%']
    lbl = cols[i] + ' (Median: ' + str(med) + ')'
    ax[0,i].hist(d[cols[i]])
    ax[0,i].axvline(med, color='r', linewidth=2)
    ax[0,i].set_xlabel(lbl)

cols = ['Epsilon', 'Alpha']
for i in range(len(cols)):
    med = d[cols[i]].describe()['50%']
    lbl = cols[i] + ' (Median: ' + str(med) + ')'
    ax[1,i].scatter(d['Total'], d[cols[i]])
    ax[1,i].axhline(med, color='r', linewidth=2)
    ax[1,i].set_ylabel(lbl)
    ax[1,i].set_xlabel('Total Steps')
    ax[1,i].yaxis.grid()
    ax[1,i].xaxis.grid()

m, b = np.polyfit(d['Total'], d['Steps'], 1)
ax[2,0].scatter(d['Total'], d['Steps'])
ax[2,0].plot(d['Total'], m*d['Total'] + b, '-', color='r', linewidth=2)
ax[2,0].set_xlabel('Total Steps')
ax[2,0].set_ylabel('Steps per Trial')
ax[2,0].yaxis.grid()
ax[2,0].xaxis.grid()

m, b = np.polyfit(d['Total'], d['Rewards'], 1)
ax[2,1].scatter(d['Total'], d['Rewards'])
ax[2,1].plot(d['Total'], m*d['Total'] + b, '-', color='r', linewidth=2)
ax[2,1].set_xlabel('Total Steps')
ax[2,1].set_ylabel('Rewards')
ax[2,1].yaxis.grid()
ax[2,1].xaxis.grid()

plt.subplots_adjust(wspace=.3)
plt.subplots_adjust(hspace=.3)

for ax in f.axes[2:]:
    plt.sca(ax)
    plt.xticks(rotation=45)

plt.show()
