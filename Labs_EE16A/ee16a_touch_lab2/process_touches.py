 #!/usr/bin/env python
import time
import sys
import serial
import argparse
# ADDED GLOB
import glob

def get_loc(x_raw, y_raw, raw_range=4096, x_range=3, y_range=3):
    """Convert raw x and y analog values to touchscreen locations.

    Args:
        x_raw: raw analog x value readin by the Launchpad.
        y_raw: raw analog y value readin by the Launchpad.
        raw_range: the span of the raw values (starting from 0 implied).
        x_range: resolution of the touchscreen in the x axis.
        y_range: resolution of the touchscreen in the y axis.

    Returns:
        tuple(x,y): x and y MUST BE INTEGERS that correspond to the (x,y) location of a touch.
    Hint: Use int( ) to cast your numbers to integers
    """
    x = 1 # YOUR CODE HERE
    y = 1 # YOUR CODE HERE
    return (x,y)

def serial_ports():
    """Lists serial ports

    Raises:
    EnvironmentError:
        On unsupported or unknown platforms
    Returns:
        A list of available serial ports
    """
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(500)]

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

def open_port(port, baud):
    """Open a serial port.

    Args:
    port (string): port to open, on Unix typically begin with '/dev/tty', on
        or 'COM#' on Windows.
    baud (int, optional): baud rate for serial communication

    Raises:
    SerialException: raised for errors from serial communication

    Returns:
       A pySerial object to communicate with the serial port.
    """
    ser = serial.Serial()
    try:
        ser = serial.Serial(port, baud, timeout=10)
        return ser
    except serial.SerialException:
        raise

def parse_serial(ser):
    """Extract the raw x and y values from Serial.

    Args:
        ser: the serial port object.

    Returns:
        tuple(int x, int y): the raw x and the raw y values read in by the Launchpad.
    """
    try:
        data = ser.readline().decode().split()
        if len(data) == 6 and data[0] == 'X' and data[3] == 'Y':
            return (int(float(data[2])), int(float(data[5])))
    except Exception as e:
        quit()


def main(device="/dev/ttyACM0", baud=9600):
    """Continually read from serial and print out touch locations.

    Args:
        device: port identifier for some serial device.
        baud: the baud rate.

    Raises:
        KeyboardInterrupt

    Returns:
        Nothing
    """
    ports = serial_ports()
    if ports:
        print("Available Ports:")
        for (i, p) in enumerate(ports):
            print("%d) %s" % (i + 1, p))
    else:
        print("No ports available. Check serial connection and try again.")
        print("Exiting...")
        quit()

    selection = input("Select the port to use: ")


    try:
        ser = serial.Serial(ports[int(selection) - 1], baud, timeout = 3)
        print("Using serial port: ", ser.port, "\n")
    except:
        print("Failed to open serial port\n")
        print("Exitting..")
        quit()

    ser.flush()
    print("Start Touching Points!\n")
    try:
        while True:
            if ser.inWaiting() > 0:
                x_raw, y_raw = parse_serial(ser)
                print("\nx_raw=%d, y_raw=%d" % (x_raw,y_raw)) # Feel free to comment this line out so it doesn't showup in the terminal.
                x, y = get_loc(x_raw, y_raw)
                print("Touched at: (%r, %r)" % (x,y))
    except KeyboardInterrupt:
        quit()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Read raw touchscreen values')
    parser.add_argument('-d', '--device', choices=serial_ports(), default="/dev/ttyACM0",
            help='a serial port to read from')
    parser.add_argument('-b', '--baud', type=int, default=9600,
            help="baud rate for the serial connection ")
    args = parser.parse_args()
    print('Touchscreen Processing Started.\n')
    main(args.device, args.baud)
