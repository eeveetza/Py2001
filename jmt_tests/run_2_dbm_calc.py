from Py2001 import P2001
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Welcome to dBm calculation using using Py2001")

    # Set inputs
    tx_power_dbm = 40.0
    tx_ant_gain_dbi = 10.0
    rx_ant_gain_dbi = 10.0
    max_range_km = 60.0
    range_interval_km= 5.0
    noise_floor = -105.0
    min_snr = 10.0

    # Set up P.2001-4 inputs
    z = np.array([1]) # 1 - Sea, 3 - Coastal Land, 4 - Inland
    GHz = 2.4
    Tpc = 90.0
    Phire = -3.070829
    Phirn = 50.703156
    Phite = -3.069410
    Phitn =  50.653696
    Hrg = 1.5
    Htg = 1.5
    Grx = rx_ant_gain_dbi
    Gtx = tx_ant_gain_dbi
    FlagVP = 1

    # Calculate P2001-4 losses for each range
    ranges = np.arange(range_interval_km, max_range_km+range_interval_km, range_interval_km)
    print(ranges)

    losses = [] 
    for range in ranges:
        print("Calculating losses for range = ", range, " km")
        # Set up terrain profile
        d = np.linspace(0.0, range, 100)
        h = np.zeros(d.shape[0], dtype=float)
        print("d = ", d)
        print("h = ", h)

        # Calculate P2001-4 losses
        Lb = P2001.bt_loss(d, h, z, GHz, Tpc, Phire, Phirn, Phite, Phitn, Hrg, Htg, Grx, Gtx, FlagVP)

        print(Lb)
        losses.append(Lb)
    
    # Convert to numpy array
    losses = np.array(losses)
    print(losses)

    # Plot the losses vs range
    plt.plot(ranges, losses, '-bo')
    plt.xlabel("Range (km)")
    plt.ylabel("P.2001-4 Losses (dB)")
    plt.show()

    # Calculate expected signal power (dBm) at each range
    rx_power_dbm = tx_power_dbm + tx_ant_gain_dbi + rx_ant_gain_dbi - losses
    print(rx_power_dbm)

    # Plot the rx power vs range
    plt.plot(ranges, rx_power_dbm, '-bo', label="Expected Rx Signal Power (dBm)")
    plt.hlines(y=noise_floor, xmin=ranges[0], xmax=ranges[-1], linestyles='--', colors='red', label="Noise Floor (dBm)")
    plt.hlines(y=noise_floor+min_snr, xmin=ranges[0], xmax=ranges[-1], linestyles='--', colors='green', label="Rx Power for Minimum SNR (dBm)")
    plt.legend(loc="upper right")
    plt.xlabel("Range (km)")
    plt.ylabel("Rx Signal Power (dBm)")
    plt.title("Rx Signal Power (dBm) vs range (km) at " + str(Tpc) + "% temporal availability" )
    plt.show()