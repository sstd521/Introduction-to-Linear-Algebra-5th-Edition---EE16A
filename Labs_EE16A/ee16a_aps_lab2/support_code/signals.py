import scipy.io



#### Uncomment the line that corresponds to the station you will use. 
#### Pre-recorded data is correlated with the A-station.
MATLAB_FILE_NAME_A = 'support_code/Adata.mat'
MATLAB_FILE_NAME_B = 'support_code/Bdata.mat'
MATLAB_FILE_NAME_C = 'support_code/Cdata.mat'
MATLAB_FILE_NAME_D = 'support_code/Ddata.mat'
####



class Signal(object):
	def __init__(self):
		self.mat_file_A = scipy.io.loadmat(MATLAB_FILE_NAME_A)
		self.mat_file_B = scipy.io.loadmat(MATLAB_FILE_NAME_B)
		self.mat_file_C = scipy.io.loadmat(MATLAB_FILE_NAME_C)
		self.mat_file_D = scipy.io.loadmat(MATLAB_FILE_NAME_D)

	def get_variable(self, var_name, stationLabel):
		if stationLabel == 'A':
			if var_name in self.mat_file_A:
			    return self.mat_file_A[var_name]
			return none
		if stationLabel == 'B':
			if var_name in self.mat_file_B:
			    return self.mat_file_B[var_name]
			return none
		if stationLabel == 'C':
			if var_name in self.mat_file_C:
			    return self.mat_file_C[var_name]
			return none
		if stationLabel == 'D':
			if var_name in self.mat_file_D:
				return self.mat_file_D[var_name]
			return none
		#if var_name in self.mat_file:
			#return self.mat_file[var_name]
		#return None

	def get_beacon(self,stationLabel):
		#stationLabel = 'A'
		#stationName = 'B'
		#stationName = 'C'
		#stationName = 'D'

		beacon0 = self.get_variable("beacon0",stationLabel)[0]
		beacon1 = self.get_variable("beacon1",stationLabel)[0]
		beacon2 = self.get_variable("beacon2",stationLabel)[0]
		beacon3 = self.get_variable("beacon3",stationLabel)[0]
		beacon4 = self.get_variable("beacon4",stationLabel)[0]
		beacon5 = self.get_variable("beacon5",stationLabel)[0]

		beacon = [beacon0, beacon1, beacon2, beacon3, beacon4, beacon5]
		return beacon

# singleton
Signal = Signal()

LPF = [
		 0.011961120715177888, 0.017898431488832207,
		 0.023671088777712977, 0.023218234511504027,
	     0.014273841944786474, -0.0019474257432360068,
	     -0.020353564773665882, -0.033621739954825786,
	     -0.03551208496295946, -0.02427977087856695,
	     -0.004505671193971414, 0.01417985329617787,
	     0.02120515509832614, 0.010524501050778973,
	     -0.01526011832898763, -0.044406684807689674,
	     -0.06003132747249487, -0.047284818566536956,
	     -0.0006384172460392303, 0.07212437221154776,
	     0.151425957854483, 0.21267819920402747,
	     0.23568789602698456, 0.21267819920402747,
	     0.151425957854483, 0.07212437221154776,
	     -0.0006384172460392303, -0.047284818566536956,
	     -0.06003132747249487, -0.044406684807689674,
	     -0.01526011832898763, 0.010524501050778973,
	     0.02120515509832614, 0.01417985329617787,
	     -0.004505671193971414, -0.02427977087856695,
	     -0.03551208496295946, -0.033621739954825786,
	     -0.020353564773665882, -0.0019474257432360068,
	     0.014273841944786474, 0.023218234511504027,
	     0.023671088777712977, 0.017898431488832207,
	     0.011961120715177888
	   ]
