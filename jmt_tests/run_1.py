from Py2001 import P2001
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Welcome to run 1 using Py2001")

    # Instantiate the inputs needed by the bt_loss() function
    d = np.array([
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0]) # units: km
    h = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0]) # units: m
    plt.plot(d, h, '-bo')
    plt.show()
    z = np.array([1]) # 1 - Sea, 3 - Coastal Land, 4 - Inland
    GHz = 2.4
    Tpc = 90.0
    Phire = 0.000000
    Phirn = 0.000000
    Phite = 0.100000
    Phitn = 0.000000
    Hrg = 1.0
    Htg = 1.0
    # Note: compute gain for parabolic antennas:
    # https://www.everythingrf.com/rf-calculators/parabolic-reflector-antenna-gain
    Grx = 10.0
    Gtx = 10.0
    FlagVP = 1

    Lb = P2001.bt_loss(d, h, z, GHz, Tpc, Phire, Phirn, Phite, Phitn, Hrg, Htg, Grx, Gtx, FlagVP)

    print("Lb = ", Lb, " dB")
    print("Tpc = ", Tpc, "% of time")
