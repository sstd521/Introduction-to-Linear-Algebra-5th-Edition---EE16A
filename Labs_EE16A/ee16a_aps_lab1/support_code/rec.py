from mic import SoundRecorder
try:
	mic = SoundRecorder()
	mic.open_stream()
except OSError as e:
	if (e.args == ('Invalid number of channels', -9998)):
		print("ERROR: Please insert the microphone into the appropriate port on the front of the computer!")
	else:
		raise