import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumtrapz
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from pylab import rcParams

def test_cap_IV(cap_IV):
    i_test = np.array([0.002,0.002,0.002,0.002,0.002])
    c_test = 1e-9
    t_test = 3e-6
    V_correct = [0.0, 6.0, 12.0, 18.0, 24.0]
    V_check = np.array(cap_IV(i_test, c_test, t_test))
    rtol = 1e-4
    if (np.linalg.norm(V_check - V_correct) / np.linalg.norm(V_correct) > rtol):
        print("Incorrect! Fix your cap_IV before moving on")
    else:
        print("cap_IV test passed!")

def square_wave_zerod():
    return np.concatenate((np.array(200*[0]), (0.01 * signal.square(2 * np.pi * 3 * np.linspace(0, 1, 1200)))))

def integrate(function, dt, c=0):
    return cumtrapz(function, dx=dt, initial=c);

def gen_vin():
    return .05 * signal.square(2 * np.pi * 2 * np.linspace(0, 1, 1000))[125:625:]

