import math
from math import log10
import numpy as np


def axis_limit_tool(lower, upper, **kwargs):
	"""
	Rounds the axis limits for the purpose of always having a grid line beyond the largest and smallest data point.
	:param lower: lower bound of yaxis from ax.get_ylim()
	:param upper: upper bound of yaxis from ax.get_ylim()
	:param kwargs: two optional keyword arguments
		- lower_decade: e.g. ...0.01, 0.1, 1, 10, 100...
		- upper_decade: e.g. ...0.01, 0.1, 1, 10, 100...
	:return: A tuple of the updated y axis limits, <(lower_new, upper_new)>
	"""

	if kwargs:
		lower_decade = kwargs.pop('lower_decade')
		upper_decade = kwargs.pop('upper_decade')
	else:
		lower_decade = None
		upper_decade = None

	# # Fix Lower Bound
	# Currently works for positive numbers
	# Always rounds to the 1s place if Greater than 1
	# If less than 1, rounds to the greatest decade
	if not lower_decade:
		lower_new = math.floor(lower)
		i = 1
		while True:
			if lower_new >= 0:  # Positive
				if lower_new == 0:
					lower_new = float(math.floor((10**i)*lower))/(10**i)
					i += 1
				else:
					break
			else:
				# Changing the method to <*-1, ceil, *-1>
				lower = lower * -1
				i = 1
				while True:
					if lower > 1:  # Larger Numbers
						lower_new = float(math.ceil(lower))
						lower_new = -1 * lower_new
						break
					else:  # Smaller Numbers
						if lower*i >= 1:
							lower_new = float(math.ceil(lower*i))/i
							lower_new = -1 * lower_new
							break
						else:  # lower*i < 1:
							i = i * 10
	else:
		lower_new = (lower // lower_decade) * lower_decade

	# # Fix Upper Bound
	if not upper_decade:
		i = 1
		while True:
			if upper*i >= 1:
				upper_new = float(math.ceil(upper*i))/i
				break
			elif upper*i < 1:
				i = i * 10
			else:
				i = i / 10
	else:
		upper_new = math.ceil(upper)

	return lower_new, upper_new


def calc_int_phase_jitter(xData, yData, fcar_hz, int_band):
	"""
	Integrated PN sideband of x_data and y_data

	Requires numpy as np

	:param xData: type: np array of floats. xData and yData must be the same dimensions. 1xN
	:param yData: type: np array of floats
	:param fcar_hz: Carrier frequency, in Hz.  type: int or float
	:param int_band: list of two ints or floats. Call function multiple times for integration across multiple bands
	:return: rms jitter in seconds
	"""
	# Find the index in xData of the value closest to each integration limit
	idxLowerLimit = np.abs(xData - int_band[0]).argmin()
	idxUpperLimit = np.abs(xData - int_band[1]).argmin()

	# Splice the data based on the limits of integration
	xData = xData[idxLowerLimit:(idxUpperLimit + 1)]
	yData = yData[idxLowerLimit:(idxUpperLimit + 1)]

	# Convert the data to linear space
	yDataLin = 10 ** (yData / 10)

	# Integrate using trapezoidal integration
	sum_jit = 0
	for i in range(0, len(yDataLin) - 1):
		avgHeight = (yDataLin[i] + yDataLin[i + 1]) / 2
		sum_jit += (xData[i + 1] - xData[i]) * avgHeight

	intPhaseJitter = 1e12 * (np.sqrt(2 * sum_jit) / (2 * np.pi * fcar_hz))

	return intPhaseJitter


def scale_pn_fcar(fcar_old, fcar_new, ydata):
	""" Scales a phase noise sideband to another carrier frequency using the formula ydata+20log10(fcar_new/fcar_old)
	:param fcar_old:
	:param fcar_new:
	:param ydata:
	:return: scaled ydata list"""

	ydata_scaled = list(map(lambda t: t + 20 * log10(fcar_new / fcar_old), ydata))
	ydata_scaled = np.array(ydata_scaled)
	return ydata_scaled


def calc_int_per_jitter(xData, yData, fcar_hz, int_band=(100, 1E9)):
	"""
	Calculates RMS period jitter from SSB phase noise PSD.

	:param xData: np array of floats. xData and yData must be the same dimensions. 1xN
	:param yData: np array of floats
	:param fcar_hz: Carrier frequency, in Hz.  type: int or float
	:param int_band: list of two ints or floats.
	:return:
	"""
	# Find the index in xData of the value closest to each integration limit
	idxLowerLimit = np.abs(xData - int_band[0]).argmin()
	idxUpperLimit = np.abs(xData - int_band[1]).argmin()

	# Splice the data based on the limits of integration
	xData = xData[idxLowerLimit:(idxUpperLimit + 1)]
	yData = yData[idxLowerLimit:(idxUpperLimit + 1)]

	# Convert the data to linear space
	yDataLin = 10 ** (yData / 10)

	# Apply the period jitter weighting factor
	yDataLinPJ = 8 * yDataLin * ((np.sin(np.pi * xData / fcar_hz)) ** 2)

	# Integrate using trapezoidal integration
	sum_jit = 0
	for i in range(0, len(yDataLin) - 1):
		avgHeight = (yDataLinPJ[i] + yDataLinPJ[i + 1]) / 2
		sum_jit += (xData[i + 1] - xData[i]) * avgHeight

	rmsPJ = 1e12 * (np.sqrt(sum_jit) / (2 * np.pi * fcar_hz))

	return rmsPJ


def smooth_pn(x, k):
	"""Apply a length-k median filter to a 1D array x.
	Boundaries are extended by repeating endpoints.
    """
	assert k % 2 == 1, "Median filter length must be odd."
	assert x.ndim == 1, "Input must be one-dimensional."
	k2 = (k - 1) // 2
	y = np.zeros((len(x), k), dtype=x.dtype)
	y[:, k2] = x
	for i in range(k2):
		j = k2 - i
		y[j:, i] = x[:-j]
		y[:j, i] = x[0]
		y[:-j, -(i+1)] = x[j:]
		y[-j:, -(i+1)] = x[-1]

	return np.median(y, axis=1)


def calc_running_integration(xData, yData, fcar_hz, int_band, step_size=100):
	"""
	Calculates running integrated jitter to complement PN PSD
	:param xData: type: np array of floats. xData and yData must be the same dimensions. 1xN
	:param yData: type: np array of floats
	:param fcar_hz: Carrier frequency, in Hz.  type: int or float
	:param int_band: list of two ints or floats.
	:param step_size: step size for running integration. increase to speed up calculation in exchange for resolution.
	:return: xdata and ydata of cumulative jitter plot.
	"""
	# Find the index in xData of the value closest to each integration limit
	idxLowerLimit = np.abs(xData - int_band[0]).argmin()
	idxUpperLimit = np.abs(xData - int_band[1]).argmin()

	# Splice the data based on the limits of integration
	xData = xData[idxLowerLimit:(idxUpperLimit + 1)]
	yData = yData[idxLowerLimit:(idxUpperLimit + 2)]

	# Convert the data to linear space
	yDataLin = 10 ** (yData / 10)

	# Integrate from start to each index  and save results in an numpy array
	yData_cumulative = np.array([])
	xData_cumulative = np.array([])
	for idx in range(1, len(xData), step_size):
		sum_jit = 0
		for i in range(0, idx):
			avgHeight = (yDataLin[i] + yDataLin[i + 1]) / 2
			sum_jit += (xData[i + 1] - xData[i]) * avgHeight
		intPhaseJitter = 1e12 * (np.sqrt(2 * sum_jit) / (2 * np.pi * fcar_hz))
		yData_cumulative = np.append(yData_cumulative, [intPhaseJitter])
		xData_cumulative = np.append(xData_cumulative, [xData[idx]])
	yData_cumulative = np.vstack(yData_cumulative)
	xData_cumulative = np.vstack(xData_cumulative)

	return xData_cumulative, yData_cumulative


def axis_limit_tool_beta(lower, upper, **kwargs):
	"""
	Rounds the axis limits for the purpose of always having a grid line beyond the largest and smallest data point.
	:param lower: lower bound of yaxis from ax.get_ylim()
	:param upper: upper bound of yaxis from ax.get_ylim()
	:param kwargs: two optional keyword arguments
		- lower_decade: e.g. ...0.01, 0.1, 1, 10, 100...
		- upper_decade: e.g. ...0.01, 0.1, 1, 10, 100...
	:return: A tuple of the updated y axis limits, <(lower_new, upper_new)>
	"""

	if kwargs:
		lower_decade = kwargs.pop('lower_decade')
		upper_decade = kwargs.pop('upper_decade')
	else:
		lower_decade = None
		upper_decade = None

	# # Fix Lower Bound
	if lower <= 0:
		lower_new = lower
	else:
		if not lower_decade:
			# Step 1: Determine Order of Magnitude
			order = math.floor(math.log10(lower))
			# Step 2: Find lower bound of that order
			lower_new = math.floor(lower/(10**order))*(10**order)

			# Ignore below, Currently works for positive num
			# i = 1
			# while True:
			# 	if lower_new >= 0:  # Positive
			# 		if lower_new == 0:
			# 			lower_new = float(math.floor((10**i)*lower))/(10**i)
			# 			i += 1
			# 		else:
			# 			break
			# 	else:
			# 		# Changing the method to <*-1, ceil, *-1>
			# 		lower = lower * -1
			# 		i = 1
			# 		while True:
			# 			if lower > 1:  # Larger Numbers
			# 				lower_new = float(math.ceil(lower))
			# 				lower_new = -1 * lower_new
			# 				break
			# 			else:  # Smaller Numbers
			# 				if lower*i >= 1:
			# 					lower_new = float(math.ceil(lower*i))/i
			# 					lower_new = -1 * lower_new
			# 					break
			# 				else:  # lower*i < 1:
			# 					i = i * 10
		else:
			lower_new = (lower // lower_decade) * lower_decade

	# # Fix Upper Bound
	if not upper_decade:
		# Step 1: Determine Order of Magnitude
		order = math.floor(math.log10(upper))
		# Step 2: Find lower bound of that order
		upper_new = math.floor(upper / (10 ** order) + 1) * (10 ** order)
	else:
		upper_new = (upper // upper_decade + 1) * upper_decade

	return lower_new, upper_new