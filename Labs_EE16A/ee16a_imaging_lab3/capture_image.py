#!/usr/bin/python
from __future__ import unicode_literals
import sys
import argparse
import numpy as np
import matplotlib
# matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import time
import glob
import serial
import struct

from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

global EE16A_SCAN_DELAY
# Total time to delay for at start of scan in milliseconds
EE16A_SCAN_DELAY = 1000

# Projector dimensions are 1280 x 720
DEFAULT_DISP_WIDTH = 1280
DEFAULT_DISP_HEIGHT = 720

BAUD_RATE = 115200

help_menu = """Quit: press [Esc], [Ctrl+Q], or [Ctrl+W] at any time to exit\n
Help: press [H] to show this help menu\n"""

def serial_ports():
  """Lists serial ports

  Raises:
  EnvironmentError:
      On unsupported or unknown platforms
  Returns:
      A list of available serial ports
  """
  if sys.platform.startswith('win'):
      ports = ['COM' + str(i + 1) for i in range(256)]

  elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
      # this is to exclude your current terminal "/dev/tty"
      ports = glob.glob('/dev/tty[A-Za-z]*')

  elif sys.platform.startswith('darwin'):
      ports = glob.glob('/dev/tty.*')

  else:
      raise EnvironmentError('Unsupported platform')

  result = []
  for port in ports:
      try:
          s = serial.Serial(port)
          s.close()
          result.append(port)
      except (OSError, serial.SerialException):
          pass
  return result

def serial_write(ser, data):
  if sys.platform.startswith('win'):
    ser.write([data, ])
  else:
    ser.write(data)

class Mask(QtGui.QWidget):
  def __init__(self, ser, fps, imgWidth, imgHeight, infile, outfile, numCaptures, brightness, sleepTime):
    super(Mask, self).__init__()

    self.setAutoFillBackground(True)
    p = self.palette()
    p.setColor(self.backgroundRole(), QtCore.Qt.black)
    self.setPalette(p)

    # Set up shortcuts to close the program
    QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.close)
    QtGui.QShortcut(QtGui.QKeySequence("Ctrl+W"), self, self.close)
    QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.close)
    QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.close)
    QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, self.close)
    QtGui.QShortcut(QtGui.QKeySequence("H"), self, self.help)

    self.ser = ser                  # Serial port
    self.scan_rate = fps            # Scan rate in fps (each pixel movement = 1 frame)
    self.imgWidth = imgWidth        # Matrix width
    self.imgHeight = imgHeight      # Matrix height
    self.mask_file = infile
    self.out_file = outfile
    self.prevTime = 0
    self.currTime = 0

    self.numCaptures = numCaptures
    self.brightness = brightness / 100
    self.sleepTime = sleepTime / 1000

    self.minTimeDelta = 1000
    self.maxTimeDelta = 0

    # Load the imaging mask
    self.Hr = np.load(self.mask_file) * 255 * self.brightness
    # Measurement rows
    self.numMeasurements = self.Hr.shape[0]
    self.currCapture = 0
    self.dispWidth = DEFAULT_DISP_WIDTH      # Display width
    self.dispHeight = DEFAULT_DISP_HEIGHT    # Display height

    self.initSelf()

  def initSelf(self):
    self.count = 0
    self.dataLoc = 0
    self.started= False
    self.fullscreen = True
    self.sensor_readings = np.zeros((self.numMeasurements, 1))
    self.time0 = 0                           # Will be used to time scan
    self.time_final = 0
    self.fp_flag = 0

    self.initUI()

  def updateProjector(self, count):
    # Need to scale mask aspect ratio accordingly.
    # Measurement "row"
    mask = np.reshape(self.Hr[count, :], (self.imgHeight, self.imgWidth))
    mask = np.require(mask, np.uint8, 'C')
    QI = QtGui.QImage(mask, mask.shape[1], mask.shape[0], QtGui.QImage.Format_Indexed8)

    # If you display a square, it doesn't actually show up as square... (narrower)
    scaledWidthFactor = 4 / 3 * self.imgWidth / self.imgHeight
    scaledHeight = self.dispHeight * 0.95

    self.label.setPixmap(QtGui.QPixmap.fromImage(QI).scaled(scaledHeight * scaledWidthFactor, scaledHeight))

  def initUI(self):
    # Choose a display before first capture
    if (self.currCapture == 0):

      print("\nDetected %d screens" % QtGui.QDesktopWidget().screenCount())
      if QtGui.QDesktopWidget().screenCount() == 1:
        print("Projector not detected. Please check the connection and try again.")
        if input("Continue? [y/n]").capitalize() != "Y":
          quit()

      print("Currently displaying on screen %d" % (QtGui.QDesktopWidget().screenNumber() + 1))

      for s in range(QtGui.QDesktopWidget().screenCount()):
        print("%d) %d x %d"%(s+1, QtGui.QDesktopWidget().screenGeometry(s).width(),
          QtGui.QDesktopWidget().screenGeometry(s).height()))

      self.screen = int(input("Select the projector screen: ")) - 1

      # Set window size and center
      self.resize(self.dispWidth, self.dispHeight)
      self.center()

      # Create a label (used to display the image)
      self.label = QtGui.QLabel(self)
      self.label.setAlignment(QtCore.Qt.AlignCenter)

      self.col = QtGui.QColor(0, 0, 0)
      self.label.setGeometry(QtCore.QRect(0, 0, self.dispWidth, self.dispHeight))

    # Flush serial, display the first frame
    serial_write(self.ser, b'9')
    self.updateProjector(0)
    time.sleep(1.0)

    # Set up the timer
    self.timer = QtCore.QTimer()
    self.showNormal()
    print("\nPeriod in msecs: ", 1000 / self.scan_rate)
    self.go()

  def center(self):
    qr = self.frameGeometry()
    cp = QtGui.QDesktopWidget().availableGeometry(self.screen).center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())
    if QtGui.QDesktopWidget().screenCount() == 2 and self.screen == 1:       # They have chosen to display on projector
      self.move(QtGui.QDesktopWidget().screenGeometry(0).width() + 12, -12)

  def readSensorData(self):
    # Tell MSP to collect data
    serial_write(self.ser, b'6')

    serial_data = self.ser.readline()
    # Convert ASCII characters from serial port to integers
    serial_data_processed = []
    for i in serial_data:
      if chr(i) != '\r' and chr(i) != '\n':
        serial_data_processed.append(i)
    serial_data_length = len(serial_data_processed)

    data = 0
    count = 1

    for j in serial_data_processed:
      data += pow(10, serial_data_length - count) * int(chr(j))
      count += 1

    # When timeout is too low, sometimes nothing is received... we want a redo in that case (detect invalid)
    if (serial_data_length == 0):
      data = -1
    return data;

  def updateData(self):
    global EE16A_SCAN_DELAY

    # Not sure why I need this... data doesn't seem to be properly aligned
    # Probably some interplay between timer interrupt + sleep?
    if (self.sleepTime > 0.09):
      shiftCorrection = 2
    else:
      shiftCorrection = 3

    if self.count < self.numMeasurements + shiftCorrection:
      if self.fp_flag < EE16A_SCAN_DELAY / (1000 / self.scan_rate):
        self.fp_flag += 1
        # Need to match capture index with screen being displayed
        self.count = -1
      elif self.fp_flag == EE16A_SCAN_DELAY / (1000 / self.scan_rate):
        self.time0 = time.time()
        self.fp_flag += 1

      # Update what's being displayed to projector
      if(self.count == -1):
        cnt = self.numMeasurements - 1
      else:
        cnt = self.count

      # Don't update screen on last scan count (just read out value)
      if (self.count < self.numMeasurements):
        self.updateProjector(cnt)

      # Sleep for some time to allow sensor output to settle
      time.sleep(self.sleepTime)

      # Looks like when UART isn't ready, it'll just show up as newline, which gets read and screws up timing
      # Alternatives: change timeout (increase, so scan takes longer) OR [what I do] just don't increment save location
      # when a blank line is returned (hopefully everything gets properly appended at the end)
      if (self.count == -1):
        self.dataLoc = -1 - shiftCorrection

      if (self.count < shiftCorrection):
        data = self.readSensorData()
        if data > -1:
          self.dataLoc += 1
      elif (self.count >= shiftCorrection):
        data = self.readSensorData()
        self.prevTime = self.currTime
        self.currTime = time.time()
        timeDelta = self.currTime - self.prevTime
        if (self.count > shiftCorrection and self.count < shiftCorrection + 5):
          print("Time delta between captures in s: %s" % timeDelta)
        elif (self.count > shiftCorrection):
          if (timeDelta > self.sleepTime * 1.35 or timeDelta < self.sleepTime * 0.65):
            print("Time delta between captures in s: %s" % timeDelta)

        if (timeDelta < self.minTimeDelta and self.count > shiftCorrection):
          self.minTimeDelta = timeDelta
        if (timeDelta > self.maxTimeDelta and self.count > shiftCorrection):
          self.maxTimeDelta = timeDelta

        self.sensor_readings[self.dataLoc] = data
        if (self.dataLoc % 100 == 0 or self.dataLoc < 10 or self.dataLoc > (self.numMeasurements + shiftCorrection - 10) or data == -1):
          print("Loc: %r Data: %r " % (self.dataLoc, data))
        if (data > -1):
          self.dataLoc += 1
        elif (data == -1):
          print("Losing data! Consider increasing your timeout!")

      # if timeout not long enough, redo!
      self.count += 1

    else:

      if (self.dataLoc) < self.numMeasurements - 1:
        print("Lost %s data! :( Please increase timeout!" % ((self.numMeasurements - 1) - self.dataLoc))

      self.timer.stop()
      self.time_final = time.time()
      print("\nScan completed")
      print("Scan time: %.3f s" % (self.time_final - self.time0))
      print("Min time delta in s: %s" % self.minTimeDelta)
      print("Max time delta in s: %s" % self.maxTimeDelta)

      extra_data = []
      while(self.ser.inWaiting()):
        data = self.readSensorData()
        extra_data.append(data)

      print("Length of extra_data: ", len(extra_data))
      print("Extra Data: ", extra_data)

      self.sensor_readings = np.append(self.sensor_readings[:self.dataLoc], np.reshape(extra_data, (len(extra_data), 1)), axis = 0)

      # If you're just taking linear measurements, don't need to restrict yourself to screen size
      # N = width
      reshapeHeight = int(np.ceil(len(self.sensor_readings) / self.imgWidth))
      reshapeSize = reshapeHeight * self.imgWidth

      # Awkwardly fill in to get a full row for displaying (why ceil is used)
      padAmount = reshapeSize - len(self.sensor_readings)

      # Should be column vector!
      paddedSensorReadings = np.append(self.sensor_readings, np.transpose([[np.amax(self.sensor_readings)] * padAmount]), axis = 0)

      if (reshapeHeight != int(len(self.sensor_readings) / self.imgWidth)):
        print("Incomplete measurement row. Is your mask matrix size correct? Padding outputs %s times for display." % padAmount)

      saveFile = self.out_file + "_" + str(int(self.brightness * 100)) + "_" + str(self.currCapture)
      print("Saving data as %s.npy" % saveFile)

      np.save("%s.npy" % saveFile, self.sensor_readings)
      self.currCapture += 1

      plt.figure()
      img = np.reshape(paddedSensorReadings, (reshapeHeight, self.imgWidth))
      plt.imshow(img, cmap='gray', interpolation="nearest")
      plt.title('Capture Image Results')
      plt.show()

      print("Min sensor reading: %s" % np.amin(paddedSensorReadings))
      print("Max sensor reading: %s" % np.amax(paddedSensorReadings))

      if (self.currCapture < self.numCaptures):
        self.initSelf()
      else:
        self.close()

  def help(self):
    self.fullscreen = True
    self.rescale()
    self.showNormal()
    print(help_menu)

  def rescale(self):
    if self.fullscreen:
      self.dispWidth = DEFAULT_DISP_WIDTH   # Display width
      self.dispHeight = DEFAULT_DISP_HEIGHT # Display height
    else:
      self.dispWidth = QtGui.QDesktopWidget().screenGeometry(self.screen).width()
      self.dispHeight = QtGui.QDesktopWidget().screenGeometry(self.screen).height()

    self.label.setGeometry(QtCore.QRect(0, 0, self.dispWidth, self.dispHeight))
    QI=QtGui.QImage(self.mask.data, self.imgWidth, self.imgHeight, QtGui.QImage.Format_Indexed8)
    self.label.setPixmap(QtGui.QPixmap.fromImage(QI).scaled(self.dispWidth, self.dispHeight, QtCore.Qt.KeepAspectRatio))

  def go(self):
    self.ser.flushOutput()
    self.ser.flushInput()
    time.sleep(1.0)
    print("\nStarting scan %s... \n" % str(self.currCapture))
    self.started=True
    self.timer.start(1000 / self.scan_rate)
    self.connect(self.timer, QtCore.SIGNAL('timeout()'), self.updateData)
    # self.timer.timeout.connect(self.updateData)

def main():
  print("\nEE16A Imaging Lab\n")

  # Parse arguments
  parser = argparse.ArgumentParser(description = 'This program projects a sampling pattern and records the corresponding phototransistor voltage.')
  parser.add_argument('-f', '--fps', type = int, default = 40, help = 'frames per second (default = 40)')
  parser.add_argument('--width', type = int, default = 40, help = 'width of the image in pixels (default = 40px)')
  parser.add_argument('--height', type = int, default = 30, help = 'height of the image in pixels (default = 30px)')
  parser.add_argument('--mask', default = "masks/diag_mask.npy", help = 'saved sampling pattern mask (default = "masks/diag_mask.npy")')
  parser.add_argument('--out', default = "images/sensor_readings", help = 'output path (default="images/sensor_readings", saved as "images/sensor_readings_0.npy")')
  parser.add_argument('--numCaptures', type = int, default = 1, help = 'number of image captures (for averaging)')
  parser.add_argument('--sleepTime', type = int, default = 120, help = 'sleep time in milliseconds -- time between projector update and data capture')
  parser.add_argument('--brightness', type = int, default = 100, help = '% of full projector brightness to use')
  parser.add_argument('--timeout', type = int, default = 150, help = 'serial timeout in ms (default = 40)')

  args = parser.parse_args()

  print("Serial timeout in ms: %d" % args.timeout)
  print("Sleep time in ms: %d" % args.sleepTime)
  print("Projector brightness scale in percent: %d" % args.brightness)

  print("FPS: %d" % args.fps)
  print("Image width: %d" % args.width)
  print("Image height: %d" % args.height)

  print("Mask file: %s \n" % args.mask)

  print("Checking serial connections...")

  ports = serial_ports()
  if ports:
    print("Available serial ports:")
    for (i, p) in enumerate(ports):
      print("%d) %s" % (i + 1, p))
  else:
    print("No ports available. Check serial connection and try again.")
    print("Exiting...")
    quit()

  timeout = args.timeout / 1000
  print("If you don't see correct COM port, enter \'0\' and close Energia, then restart this code block")
  selection = input("Select the port to use: ")

  if selection == '0':
    return

  # Note: seems like timeout of 1 doesn't work
  ser = serial.Serial(ports[int(selection) - 1], BAUD_RATE, timeout = timeout)
  app = QtGui.QApplication(sys.argv)
  mask = Mask(ser, args.fps, args.width, args.height, args.mask, args.out, args.numCaptures, args.brightness, args.sleepTime)

  sys.exit(app.exec_())

if __name__ == '__main__':
  main()
