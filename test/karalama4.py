from models.log_r import *

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = add_column_ones_design_matrix(x)
print(x)
print(y)
