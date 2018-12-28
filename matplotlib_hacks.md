
## [Seaborn Multigrid tut](https://seaborn.pydata.org/tutorial/axis_grids.html)

## Found best setting
```python

from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

```

### Check available style in matplotlib ==>> Use this `fivethirtyeight`

	plt.style.available[:10]

### Set style for entire session
	plt.style.set('stylename')

OR
with plt.style.context('stylename'):
	make_a_plot()



