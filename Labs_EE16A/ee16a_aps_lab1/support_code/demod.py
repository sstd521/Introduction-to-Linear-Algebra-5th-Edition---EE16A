from __future__ import division
v = 340.29
sampling_rate = 44100
carrier_freq = 12000
up_sample_rate = 20

import numpy
from math import sin, cos, pi, sqrt
from signals import Signal, LPF


beacon0 = Signal.get_variable("beacon0")[0]
beacon1 = Signal.get_variable("beacon1")[0]
beacon2 = Signal.get_variable("beacon2")[0]
beacon3 = Signal.get_variable("beacon3")[0]
beacon4 = Signal.get_variable("beacon4")[0]
beacon5 = Signal.get_variable("beacon5")[0]

beacon = [beacon0, beacon1, beacon2, beacon3, beacon4, beacon5]

def demodulate_signal(signal):
	"""
	Demodulate the signal using complex demodulation.
	"""
	# Demodulate the signal using cosine and sine bases
	demod_real_base = [cos(2 * pi * carrier_freq * i / sampling_rate)
		for i in range(1, len(signal) + 1)]
	demod_imaginary_base = [sin(2 * pi * carrier_freq * i / sampling_rate)
		for i in range(1, len(signal) + 1)]
	# Multiply the bases to the signal received
	demod_real = [demod_real_base[i] * signal[i] for i in range(len(signal))]
	demod_imaginary = [demod_imaginary_base[i] * signal[i] for i in range(len(signal))]
	# Filter the signals
	demod_real = numpy.convolve(demod_real, LPF)
	demod_imaginary = numpy.convolve(demod_imaginary, LPF)

	return numpy.asarray(demod_real) + numpy.asarray(demod_imaginary * 1j)

def average_signal(signal):
	beacon_length = len(beacon[0])
	num_repeat = len(signal) // beacon_length
	signal_reshaped = signal[0 : num_repeat * beacon_length].reshape((num_repeat, beacon_length))
	averaged = numpy.mean(numpy.abs(signal_reshaped), 0).tolist()
	return averaged
