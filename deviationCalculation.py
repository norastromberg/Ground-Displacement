from Sentinel2.ground_displacement_S2 import get_matrices_for_least_squares_s2
from leastSquares import least_squares
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar



def calculate_deviation():
    #de_s2,dn_s2 = get_matrices_for_least_squares_s2()
    dv,de_ls,dn_ls,ew_s2,ns_s2 = least_squares()
    return ew_s2-de_ls,ns_s2-dn_ls

def plot_deviation(de,dn):
    fig1, ax1 = plt.subplots()
    caxis1 = ax1.matshow(de, cmap=plt.get_cmap("RdBu_r"), vmin=-1, vmax=1)
    clb1 = fig1.colorbar(caxis1)
    clb1.ax.set_title('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(0, 4660, 5), ["-117.84", "-117.71", "-117.58", "-117.45", "-117.32"])
    plt.yticks(np.linspace(0, 4420, 5), ["35.91", "35.81", "35.71", "35.62", "35.52"])
    ax1.xaxis.set_ticks_position('bottom')  # the rest is the same
    ax1.set(xlabel="Longitude", ylabel="Latitude")
    ax1.set_title("Deviation East-West", weight="bold")
    scalebar = ScaleBar(10, location="lower left")  # 1 pixel = 10 meter
    plt.gca().add_artist(scalebar)

    fig2, ax2 = plt.subplots()
    caxis2 = ax2.matshow(dn, cmap=plt.get_cmap("RdBu_r"), vmin=-1, vmax=1)
    clb2 = fig2.colorbar(caxis2)
    clb2.ax.set_title('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(0, 4660, 5), ["-117.84", "-117.71", "-117.58", "-117.45", "-117.32"])
    plt.yticks(np.linspace(0, 4420, 5), ["35.91", "35.81", "35.71", "35.62", "35.52"])
    ax2.xaxis.set_ticks_position('bottom')  # the rest is the same
    ax2.set(xlabel="Longitude", ylabel="Latitude")
    ax2.set_title("Deviation North-South", weight="bold")
    scalebar = ScaleBar(10, location="lower left")  # 1 pixel = 10 meter
    plt.gca().add_artist(scalebar)

    plt.show()


def main():
    de,dn = calculate_deviation()
    print("de-average : ", de.mean())
    print("dn-average : ", dn.mean())
    print("de-std : ", de.std())
    print("dn-std : ", dn.std())
    plot_deviation(de, dn)



if __name__ == '__main__':
   main()

