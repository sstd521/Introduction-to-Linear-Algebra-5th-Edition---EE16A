import numpy as np

def row_swap(matrix, row1, row2):
    temp = np.copy(matrix[row1][:])
    matrix[row1][:] = matrix[row2][:]
    matrix[row2][:] = temp
    return matrix

def normalize(matrix, ind):
    matrix[ind[0]][:] = matrix[ind[0]][:] / matrix[ind[0]][ind[1]]
    return matrix

def row_eliminate(matrix, ind):
    numrow, numcol = matrix.shape
    
    for row in range(numrow):
        if row != ind[0]:
            matrix[row][:] -= matrix[ind[0]][:] * matrix[row][ind[1]]
    return matrix

def gauss_elim(matrix):
    current_loc = np.array([1, 1])
    numrow, numcol = matrix.shape
    location_matrix = np.zeros([numrow, numcol]) #matrix of all zeros, and a one at the current location

    while current_loc[0] <= numrow and current_loc[1] <= numcol:
        current_col_bottom = matrix[current_loc[0]-1:,current_loc[1]-1] #current_col_bottom excludes the portion of the current column above current row
        max_col_ind = np.argmax(np.absolute(current_col_bottom))
        matrix = row_swap(matrix, current_loc[0]-1, max_col_ind+current_loc[0]-1)
        current_loc_zero = current_loc - [1,1]  # 0 indexed current location 

        if matrix[current_loc_zero[0]][current_loc_zero[1]] != 0:
            matrix = normalize(matrix,current_loc_zero)
            matrix = row_eliminate(matrix,current_loc_zero)
            current_loc += np.array([1, 1])
        else:
            current_loc[1] += np.array([1])
    return matrix