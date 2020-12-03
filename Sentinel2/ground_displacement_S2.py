from scipy.signal import detrend
import tifffile as tiff
import matplotlib.pyplot as plt
from Sentinel1.ground_displacement import plot_displacement

def read_displacement_tif(filepath):
    img = tiff.imread(filepath)

    return img


def detrending(matrix):

    return detrend(matrix)


def convert_to_metric(EW_matrix, NS_matrix):

    return EW_matrix*10, NS_matrix*(-10)


def get_matrices_for_least_squares_s2():
    rows = 4420
    cols = 4660
    min_lat, max_lat, min_long, max_long = [35.517, 35.913, -117.841, -117.323]
    imageEW = detrending(read_displacement_tif("Sentinel2/data/EW_displacement_optical.tif"))
    imageNS = detrending(read_displacement_tif("Sentinel2/data/NS_displacement_optical.tif"))

    EW_displacement, NS_displacement = convert_to_metric(imageEW, imageNS)

    return rows, cols, min_lat, max_lat, min_long, max_long, EW_displacement, NS_displacement


def main():
    rows = 4420
    cols = 4660
    imageEW = detrending(read_displacement_tif("data/EW_displacement_optical.tif"))
    imageNS = detrending(read_displacement_tif("data/NS_displacement_optical.tif"))

    EW_displacement, NS_displacement = convert_to_metric(imageEW, imageNS)
    plot_displacement(rows, cols, EW_displacement, NS_displacement, -2, 2, -2, 2, "East-West displacement", "North-South displacement", 12)
    plt.show()

if __name__ == '__main__':
    main()