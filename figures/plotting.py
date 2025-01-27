'''
Plotting tools
'''

def prepare_panel(*ax):
    for axis in ax:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
