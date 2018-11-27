from __future__ import division

from signals import Signal
from math import sin, cos, pi, sqrt
from random import random
from numpy import roll
import numpy

beacon0 = Signal.get_variable("beacon0",'A')[0]
beacon1 = Signal.get_variable("beacon1",'A')[0]
beacon2 = Signal.get_variable("beacon2",'A')[0]
beacon3 = Signal.get_variable("beacon3",'A')[0]
beacon4 = Signal.get_variable("beacon4",'A')[0]
beacon5 = Signal.get_variable("beacon5",'A')[0]

beacon = [beacon0, beacon1, beacon2, beacon3, beacon4, beacon5]
beacons = beacon

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
    result_lib = numpy.convolve(signal_one, signal_two[::-1])
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
        signal_two = [1, 2]
        test = list_eq(cross_correlation(signal_one, signal_two), numpy.convolve(signal_one, signal_two[::-1]))
        if not test:
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

    # 2. Identify peaks
    if test_num == 2:
        test1 = identify_peak(numpy.array([1, 2, 2, 199, 23, 1])) == 3
        test2 = identify_peak(numpy.array([1, 2, 5, 7, 12, 4, 1, 0])) == 4
        if not (test1 and test2):
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

    # 3. Virtual Signal
    if test_num == 3:
        transmitted = beacon[0] + roll(beacon[1], 103) + roll(beacon[2], 336)
        offsets = arrival_time(beacon[0:3], transmitted)
        test = (offsets[0] - offsets[1]) == 103 and (offsets[0] - offsets[2]) == 336
        if not test:
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

def plot_speakers(plt, coords, distances, xlim=None, ylim=None, circle=True):
    """Plots speakers and circles indicating distances on a graph.
    coords: List of x, y tuples
    distances: List of distance from center of circle"""
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    xs, ys = zip(*coords)
    fig = plt.gcf()

    for i in range(len(xs)):
    	# plt.scatter(xs[i], ys[i], marker='x', color=colors[i], label='Speakers')
    	plt.scatter(xs[i], ys[i], marker='x', color=colors[i])	

    
    if circle==True:
        for i, point in enumerate(coords):
            fig.gca().add_artist(plt.Circle(point, distances[i], facecolor='none',
                                            ec = colors[i]))
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.axis('equal')
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
