### This script tests the reader module
### 
### Christopher Phillips
### github: sodoesaburningbus
### email: christopher.phillips1@valpo.edu

##### START OPTIONS #####

# Path to level 1 test file
level1_file = 'test_data/2024-08-02_00-04-08_lv1.csv'

# Path to level 2 test file
level2_file = 'test_data/2024-08-02_00-04-08_lv2.csv'

#####  END OPTIONS  #####

### Import reader
import readers
import plotters

# Test reader 1
print('Testing reader 1...')
data = readers.read_level_1(level1_file)
print('Data shape', data.shape)
print('First three rows:\n', data[:3])

print('------------------------------------')

# Test reader 2
print('Testing reader 2...')
data, heights = readers.read_level_2(level2_file)
print('Data shape', data.shape)
print('First three rows:\n', data[:3])
print('Heights:\n', heights)

# Test the bands plotter
plotters.plot_bands(data, level1_file)

# Test temperature plotter
plotters.plot_temperature(data, heights, level2_file, zmax=2)