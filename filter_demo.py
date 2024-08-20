##### START OPTIONS #####

# Path to level 1 test file
level1_file = 'test_data/2024-08-02_00-04-08_lv1.csv'

# Path to level 2 test file
level2_file = 'test_data/2024-08-02_00-04-08_lv2.csv'

#####  END OPTIONS  #####

### Import modules
import matplotlib.pyplot as pp

import readers
import tools

# Read and filter level 1 data
data = readers.read_level_1(level1_file)
filtered_data1, qc1 = tools.filter_level_1(data, sigma=1.0)
filtered_data2, qc1 = tools.filter_level_1(data, sigma=1.5)
filtered_data3, qc1 = tools.filter_level_1(data, sigma=2.0)
filtered_data4, qc1 = tools.filter_level_1(data, sigma=2.5)
filtered_data5, qc1 = tools.filter_level_1(data, sigma=3.0)

# Plot both
fig, axes = pp.subplots(ncols=2, nrows=3)

# Add each band
for i in range(9, 44):
    axes[0,0].plot(data[:,i], label=f'Band {i-8}')
    axes[1,0].plot(filtered_data1[:,i], label=f'Band {i-8}')
    axes[2,0].plot(filtered_data2[:,i], label=f'Band {i-8}')
    axes[0,1].plot(filtered_data3[:,i], label=f'Band {i-8}')
    axes[1,1].plot(filtered_data4[:,i], label=f'Band {i-8}')
    axes[2,1].plot(filtered_data5[:,i], label=f'Band {i-8}')

    axes[0,0].set_title('Unfiltered')
    axes[1,0].set_title('Filtered - 1.0 Sigma')
    axes[2,0].set_title('Filtered - 1.5 Sigma')
    axes[0,1].set_title('Filtered - 2.0 Sigma')
    axes[1,1].set_title('Filtered - 2.5 Sigma')
    axes[2,1].set_title('Filtered - 3.0 Sigma')

pp.show()