 #!/usr/bin/env python
import time
import sys
import serial
import argparse
# ADDED GLOB AND TKINTER
import glob
from tkinter import *
import random

def get_loc(x_raw, y_raw, raw_range=4096, x_range=3, y_range=3):
    """Convert raw x and y analog values to touchscreen locations.

    Args:
        x_raw: raw analog x value read-in by the Launchpad.
        y_raw: raw analog y value read-in by the Launchpad.
        raw_range: the span of the raw values (starting from 0 implied).
        x_range: resolution of the touchscreen in the x axis.
        y_range: resolution of the touchscreen in the y axis.

    Returns:
        tuple(x,y): x and y MUST BE INTEGERS that correspond to the (x,y) location of a touch.
    Hint: Use int(_) to cast your numbers to integers
    """
    x = 1 # YOUR CODE HERE
    y = 1 # YOUR CODE HERE
    return (random.randint(0,2), random.randint(0,2))

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


def start_touchscreen(baud=9600):
    """Continually read from serial and display touch locations.

    Args:
        baud: the baud rate.

    Raises:
        KeyboardInterrupt and TclError

    Returns:
        Nothing
    """
    print('Touchscreen Processing Started.\n')
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

    # GUI CONSTANTS
    on  = "#00ff00"
    off = "#a6a6a6"
    vec = {(0,0):0, (0,1):1, (0,2):2, (1,0):3, (1,1):4, (1,2):5, (2,0):6, (2,1):7, (2,2):8}
    color = [off, off, off, off, off, off, off, off, off]

    # TKINTER ROOT
    root = Tk()

    # DRAW FRAMES
    main = Frame(root)

    toprow = Frame(root)
    toprow.pack(side=BOTTOM)

    midrow = Frame(root)
    midrow.pack(side=BOTTOM)

    botrow = Frame(root)
    botrow.pack(side=BOTTOM)

    # TOP ROW
    pixel00 = Frame(toprow)
    pixel00.pack(side=LEFT)
    button00 = Button(pixel00, text="0,0", fg="grey", bg=color[0], state=DISABLED)
    button00.pack()

    pixel01 = Frame(toprow)
    pixel01.pack(side=LEFT)
    button01 = Button(pixel01, text="0,1", fg="grey", bg=color[1], state=DISABLED)
    button01.pack()

    pixel02 = Frame(toprow)
    pixel02.pack(side=LEFT)
    button02 = Button(pixel02, text="0,2", fg="grey", bg=color[2], state=DISABLED)
    button02.pack()

    # MID ROW
    pixel10 = Frame(midrow)
    pixel10.pack(side=LEFT)
    button10 = Button(pixel10, text="1,0", fg="green", bg=color[3], state=DISABLED)
    button10.pack()

    pixel11 = Frame(midrow)
    pixel11.pack(side=LEFT)
    button11 = Button(pixel11, text="1,1", fg="green", bg=color[4], state=DISABLED)
    button11.pack()

    pixel12 = Frame(midrow)
    pixel12.pack(side=LEFT)
    button12 = Button(pixel12, text="1,2", fg="grey", bg=color[5], state=DISABLED)
    button12.pack()

    # BOT ROW
    pixel20 = Frame(botrow)
    pixel20.pack(side=LEFT)
    button20 = Button(pixel20, text="2,0", fg="grey", bg=color[6], state=DISABLED)
    button20.pack()

    pixel21 = Frame(botrow)
    pixel21.pack(side=LEFT)
    button21 = Button(pixel21, text="2,1", fg="grey", bg=color[7], state=DISABLED)
    button21.pack()

    pixel22 = Frame(botrow)
    pixel22.pack(side=LEFT)
    button22 = Button(pixel22, text="2,2", fg="grey", bg=color[8], state=DISABLED)
    button22.pack()

    while True:
        try:
            if ser.inWaiting() > 0:
                x_raw, y_raw = parse_serial(ser)
                # print("\nx_raw=%d, y_raw=%d" % (x_raw,y_raw)) # Feel free to comment this line out so it doesn't showup in the terminal.
                #x, y = get_loc(x_raw, y_raw)
                #print("Touched at: (%r, %r)" % (x,y))

                color = [off, off, off, off, off, off, off, off, off]
                try:
                    color[vec[get_loc(x_raw, y_raw)]] = on
                except KeyError:
                    print("ERROR: Your get_loc function needs attention!")
                    root.destroy()
                    break

                button00.configure(bg=color[0])
                button01.configure(bg=color[1])
                button02.configure(bg=color[2])
                button10.configure(bg=color[3])
                button11.configure(bg=color[4])
                button12.configure(bg=color[5])
                button20.configure(bg=color[6])
                button21.configure(bg=color[7])
                button22.configure(bg=color[8])
                root.update()
                root.update_idletasks()
        except:
            root.destroy()
            break
