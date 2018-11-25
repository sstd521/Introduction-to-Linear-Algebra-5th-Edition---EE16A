from __future__ import division

from signals import Signal
from math import sin, cos, pi, sqrt
from random import random
from numpy import roll
import numpy
from demod import demodulate_signal

beacon0 = Signal.get_variable("beacon0")[0]
beacon1 = Signal.get_variable("beacon1")[0]
beacon2 = Signal.get_variable("beacon2")[0]
beacon3 = Signal.get_variable("beacon3")[0]
beacon4 = Signal.get_variable("beacon4")[0]
beacon5 = Signal.get_variable("beacon5")[0]

beacon = [beacon0, beacon1, beacon2, beacon3, beacon4, beacon5]

v = 340.29
sampling_rate = 44100
carrier_freq = 12000
beacon_length = 511
up_sample_rate = 20
signal_length = len(beacon0)

# The location of the speakers
speakers = [(0, 0), (5, 0), (0, 5), (5, 5), (0, 2.5), (2.5, 0)]
x0, y0 = 0, 0
x1, y1 = 5, 0
x2, y2 = 0, 5
x3, y3 = 5, 5

RANDOM_OFFSET = True # Set to False so the signals are in phase.

def generate_carrier_with_random_offset():
	rand = random()
	carrier_sample = (2 * pi *
		(carrier_freq * sample / sampling_rate + rand)
		for sample in range(1, signal_length + 1))
	return [cos(sample) for sample in carrier_sample]

def generate_carrier():
	carrier_sample = (2 * pi *
		carrier_freq * sample / sampling_rate
		for sample in range(1, signal_length + 1))
	return [cos(sample) for sample in carrier_sample]

def modulate_signal(signal, carrier):
	"""
	Modulate a given signal. The length of both signals MUST
	be the same.
	"""
	return [signal[i] * carrier[i] for i in range(signal_length)]

# Modulate beacon signals
if RANDOM_OFFSET:
	modulated_beacon0 = modulate_signal(beacon0,
		generate_carrier_with_random_offset())
	modulated_beacon1 = modulate_signal(beacon1,
		generate_carrier_with_random_offset())
	modulated_beacon2 = modulate_signal(beacon2,
		generate_carrier_with_random_offset())
	modulated_beacon3 = modulate_signal(beacon3,
		generate_carrier_with_random_offset())
	modulated_beacon = [modulate_signal(b,
		generate_carrier_with_random_offset()) for b in beacon]
else:
	modulated_beacon0 = modulate_signal(beacon0,
		generate_carrier())
	modulated_beacon1 = modulate_signal(beacon1,
		generate_carrier())
	modulated_beacon2 = modulate_signal(beacon2,
		generate_carrier())
	modulated_beacon3 = modulate_signal(beacon3,
		generate_carrier())
	modulated_beacon = [modulate_signal(b,
		generate_carrier()) for b in beacon]

def simulate_by_sample_offset(offset):
	offset1 = offset[1]
	offset2 = offset[2]
	offset3 = offset[3]
	shifted0 = modulated_beacon0 * 10
	shifted1 = modulated_beacon1 * 10
	shifted2 = modulated_beacon2 * 10
	shifted3 = modulated_beacon3 * 10
	shifted1 = roll(shifted1, offset1)
	shifted2 = roll(shifted2, offset2)
	shifted3 = roll(shifted3, offset3)
	shifted = [roll((modulated_beacon[i] * 10), offset[i]) for i in range(len(offset))]

	superposed = [shifted0[i] + shifted1[i] + shifted2[i]
		+ shifted3[i] for i in range(signal_length)]

	superposed = shifted[0]
	for i in range(1, len(shifted)):
		for j in range(len(shifted[0])):
			superposed[j] += shifted[i][j]
	return superposed

def simulate_by_location(x, y):
	distance = [sqrt((x - sp[0]) ** 2 + (y - sp[1]) ** 2) for sp in speakers]

	t_diff = [(d - distance[0]) / v for d in distance]

	# Convert to the delay / advance in sample
	# positive offset = delay, negative offset = advance

	sample_offset = [int(t * sampling_rate) for t in t_diff]
	return simulate_by_sample_offset(sample_offset)

def add_random_noise(signal, intensity):
	"""
	Add noise to a given signal.
	Intensity: the Noise-Signal Ratio.
	"""
	if intensity == 0:
		return signal
	average = sum(signal[0:100000]) / 100000
	for i in range(len(signal)):
		signal[i] = signal[i] + random() * intensity
	return signal

def get_signal_virtual(**kwargs):
	if 'intensity' not in kwargs:
		intensity = 0
	else:
		intensity = kwargs['intensity']
	if 'x' in kwargs and 'y' in kwargs:
		x = kwargs['x']
		y = kwargs['y']
		return add_random_noise(simulate_by_location(x, y), intensity)
	elif 'offsets' in kwargs:
		offsets = kwargs['offsets']
		return add_random_noise(simulate_by_sample_offset(
			offsets), intensity)
	elif 'offset' in kwargs:
		offsets = kwargs['offset']
		return add_random_noise(simulate_by_sample_offset(
			offsets), intensity)
	else:
		raise Exception("Undefined action. None is returned.")
		return None

def test_correlation(cross_correlation, signal_one, signal_two):
#    result_lib = numpy.convolve(signal_one, signal_two[::-1])
    result_lib = numpy.array([numpy.correlate(signal_one, numpy.roll(signal_two, k)) for k in range(len(signal_two))])
    result_stu = cross_correlation(signal_one, signal_two)
    return result_lib, result_stu

def test(cross_correlation, identify_peak, arrival_time, test_num):
    # Virtual Test

    # Utility Functions
    def list_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if lst1[i] != lst2[i]: return False
        return True

    test_cases = {1: "Cross-correlation", 2: "Identify peaks", 3: "Arrival time"}

    # 1. Cross-correlation function
    # If you tested on the cross-correlation section, you should pass this test
    if test_num == 1:
        signal_one = [1, 4, 5, 6, 2]
        signal_two = [1, 2, 0, 1, 2]
        test = list_eq(cross_correlation(signal_one, signal_two), numpy.convolve(signal_one, signal_two[::-1]))
        if not test:
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

    # 2. Identify peaks
    if test_num == 2:
        test1 = identify_peak(numpy.array([1, 2, 2, 199, 23, 1])) == 3
        test2 = identify_peak(numpy.array([1, 2, 5, 7, 12, 4, 1, 0])) == 4
        your_result1 = identify_peak(numpy.array([1, 2, 2, 199, 23, 1]))
        your_result2 = identify_peak(numpy.array([1, 2, 5, 7, 12, 4, 1, 0]))
        if not (test1 and test2):
            print("Test {0} {1} Failed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
    # 3. Virtual Signal
    if test_num == 3:
        transmitted = roll(beacon[0], 10) + roll(beacon[1], 103) + roll(beacon[2], 336)
        offsets = arrival_time(beacon[0:3], transmitted)
        test = (offsets[0] - offsets[1]) == (103-10) and (offsets[0] - offsets[2]) == (336-10)
        your_result1 = (offsets[0] - offsets[1])
        your_result2 = (offsets[0] - offsets[2])
        if not test:
            print("Test {0} {1} Failed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))

def plot_speakers(plt, coords, distances, xlim=None, ylim=None):
    """Plots speakers and circles indicating distances on a graph.
    coords: List of x, y tuples
    distances: List of distance from center of circle"""
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    xs, ys = zip(*coords)
    fig = plt.gcf()
    plt.scatter(xs, ys, marker='x', color='b', label='Speakers')
    for i, point in enumerate(coords):
        fig.gca().add_artist(plt.Circle(point, distances[i], facecolor='none',
                                        ec = colors[i]))
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.axis('equal')
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)


def test_identify_offsets(identify_offsets, separate_signal, average_sigs):
	# Utility Functions
	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True

	def list_sim(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 3: return False
	    return True

	test_num = 0

	# 1. Identify offsets - 1
	print(" ------------------ ")
	test_num += 1
	test_signal = get_signal_virtual(offsets = [0, 254, 114, 22, 153, 625])
	raw_signal = demodulate_signal(test_signal)
	sig = separate_signal(raw_signal)
	avgs = average_sigs(sig)
	offsets = identify_offsets(avgs)
	test = list_sim(offsets, [0, 254, 114, 23, 153, 625])
	print("Test positive offsets")
	print("Your computed offsets = {}".format(offsets))
	print("Correct offsets = {}".format([0, 254, 114, 23, 153, 625]))
	if not test:
	    print(("Test {0} Failed".format(test_num)))
	else:
	    print("Test {0} Passed".format(test_num))

	# 2. Identify offsets - 2
	print(" ------------------ ")
	test_num += 1
	test_signal = get_signal_virtual(offsets = [0, -254, 0, -21, 153, -625])
	raw_signal = demodulate_signal(test_signal)
	sig = separate_signal(raw_signal)
	avgs = average_sigs(sig)
	offsets = identify_offsets(avgs)
	test = list_sim(offsets, [0, -254, 0, -21, 153, -625])
	print("Test negative offsets")
	print("Your computed offsets = {}".format(offsets))
	print("Correct offsets = {}".format([0, -254, 0, -21, 153, -625]))
	if not test:
	    print("Test {0} Failed".format(test_num))
	else:
	    print("Test {0} Passed".format(test_num))

def test_offsets_to_tdoas(offsets_to_tdoas):
	# 3. Offsets to TDOA

	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True

	print(" ------------------ ")
	test_num = 1
	off2t = offsets_to_tdoas([0, -254, 0, -21, 153, -625], 44100)
	test = list_float_eq(numpy.around(off2t,6), numpy.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6))
	print("Test TDOAs")
	print("Your computed TDOAs = {}".format(numpy.around(off2t,6)))
	print("Correct TDOAs = {}".format(numpy.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6)))
	if not test:
	    print("Test Failed")
	else:
	    print("Test Passed")

def test_signal_to_distances(signal_to_distances):
	def list_float_eq(lst1, lst2):
	    if len(lst1) != len(lst2): return False
	    for i in range(len(lst1)):
	        if abs(lst1[i] - lst2[i]) >= 0.00001: return False
	    return True
	# 4. Signal to distances
	print(" ------------------ ")
	test_num = 1
	dist = signal_to_distances(demodulate_signal(get_signal_virtual(x=1.765, y=2.683)), 0.009437530220245524)
	test = list_float_eq(numpy.around(dist,1), numpy.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1))
	print("Test computed distances")
	print("Your computed distances = {}".format(numpy.around(dist,1)))
	print("Correct distances = {}".format(numpy.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1)))
	if not test:
	    print("Test Failed")
	else:
	    print("Test Passed")
