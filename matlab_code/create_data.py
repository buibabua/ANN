import numpy as np
import pandas as pd
import pickle
def read_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split()  # Split the line into individual elements
            matrix.append(row)  # Append the row to the matrix
    return matrix

# well data
filename = 'DD_BII.1.10_4inputs_Fixed.txt'  # Replace with your file name
matrix = read_file(filename)
well_data=pd.DataFrame(matrix,columns=['Zp','VpVs ratio','Lithology','Porosity'])
well_data.to_csv('data2.csv',index=False)
# rpt data
rpt_data=np.array(read_file('DD_RPT_BII.1-10_points.txt'),dtype='float64')
Zp_rpt=rpt_data[:,0]
VpVs_rpt=rpt_data[:,1]
y=rpt_data[:,2]
Por_V=rpt_data[:,3]
my_var={'zp':Zp_rpt,'VpVs':VpVs_rpt,'actan':y,'Porosity':Por_V}
with open('RPT_variables.pickle', 'wb') as f:
    pickle.dump(my_var, f)

# mesh rpt
x1_min, x1_max = np.min(Zp_rpt) , np.max(Zp_rpt)
x2_min, x2_max = np.min(VpVs_rpt) , np.max(VpVs_rpt)

# xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max+0.01, 50), np.arange(x2_min-0.05, x2_max, 0.005))
xx1, xx2 = np.meshgrid(np.arange(3500, 16000, 50), np.arange(1.5, 2.8, 0.01))
with open('output_grad_cal.pickle','wb') as f:
    pickle.dump({'xx':xx1,'yy':xx2},f)