import pyaudio
import struct
import math

INITIAL_TAP_THRESHOLD = 0.010
FORMAT = pyaudio.paInt16
CHANNELS = 1


RATE = 44100
INPUT_BLOCK_TIME = 2
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

class SoundRecorder(object):
	def __init__(self):
		self.audio = pyaudio.PyAudio()
		self.stream = self.open_stream()

	def open_stream(self):
		device_index=self.get_input_device()
		stream = self.audio.open(format = FORMAT,
			channels = CHANNELS, rate = RATE, input = True,
			input_device_index = device_index,
			frames_per_buffer = INPUT_FRAMES_PER_BLOCK)
		return stream

	def close_stream(self):
		self.stream.close()

	def get_input_device(self):
		return 1
		device = None
		for i in range(self.audio.get_device_count()):
			device_info = self.audio.get_device_info_by_index(i)
			print("Device%d:%s"%(i,device_info["name"]))
			for keyword in["Microph","mic","input"]:
				if keyword in device_info["name"].lower():
					print("Found an input: device%d - %s" % (i, device_info["name"]))
					device = i
					return device
		if not device:
			raise Exception("Input device not found")
		return -1

	def get_recorded_data_once(self):
		try:
			data = self.stream.read(INPUT_FRAMES_PER_BLOCK)
		except IOError as e:
			print("An error occurred when fetching data from buffer...", e)
			return None
		return data

	def get_data_chunk(self):
		err_count = 0
		for _ in range(100):
			data = self.get_recorded_data_once()
			if not data:
				err_count += 1
			else:
				print("Successfully fetched data from buffer. Error count: {0}".format(err_count))
				return data
		print("Error: failed to fetch data")
		return None

	def decode_data(self, chunk):
		num_points = len(chunk) // 2
		format = "%dh" % (num_points)
		shorts = struct.unpack(format, chunk)
		return shorts

	def get_data(self):
		raw_data = self.get_data_chunk()
		data = self.decode_data(raw_data)
		return data

	def new_data(self):
		data = self.get_recorded_data_once()
		return self.decode_data(data)


