from PIL import Image
import numpy as np
from scipy.interpolate import RectBivariateSpline
import tifffile as tiff
import matplotlib.pyplot as plt

def read_displacement_tif(filepath):
    img = tiff.imread(filepath)
    print(img.shape)
    print(type(img))
    return img

def create_coordinate_vectors(min_lat, min_long, max_lat, max_long, rows, cols):
    long_vector = np.linspace(min_long, max_long, cols)
    lat_vector = np.linspace(min_lat, max_lat, rows)

    return lat_vector, long_vector


def interpolate_matrix(matrix_to_interpolate, rows, cols, lat_min, lat_max, long_min, long_max, lat_vector, long_vector):
    x = np.linspace(long_min, long_max, cols)
    y = np.linspace(lat_min, lat_max, rows)
    interp_spline = RectBivariateSpline(y, x, matrix_to_interpolate)
    interpolated_matrix = interp_spline(lat_vector, long_vector)

    return interpolated_matrix


def plot_displacement(displacement_EW,displacement_NS):
    fig1, ax1 = plt.subplots()
    caxis1 = ax1.matshow(displacement_EW, cmap=plt.get_cmap("RdBu"))
    fig1.colorbar(caxis1)
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.5165", "35.6158", "35.7151", "35.8144", "35.9137"])
    ax1.set(xlabel="Longitude", ylabel="Latitude")
    ax1.set_title("Displacement EW")

    fig2, ax2 = plt.subplots()
    caxis2 = ax2.matshow(displacement_NS, cmap=plt.get_cmap("RdBu"))
    fig2.colorbar(caxis2)
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.5165", "35.6158", "35.7151", "35.8144", "35.9137"])
    ax2.set(xlabel="Longitude", ylabel="Latitude")
    ax2.set_title("Displacement NS")

    plt.show()


def main():
    min_lat_s2 = 35.155
    max_lat_s2 = 36.14
    min_long_s2 = -118.112
    max_long_s2 = -116.893
    rows = 10980
    cols = 10980
    min_lat, max_lat, min_long, max_long = [35.51650381422822, 35.91371799195127, -117.84199673504028, -117.322243432962]


    imageEW = read_displacement_tif("data/Px1_Num6_DeZoom1_LeChantier.tif")
    imageNS = read_displacement_tif("data/Px2_Num6_DeZoom1_LeChantier.tif")
    lat_vector, long_vector = create_coordinate_vectors(min_lat, min_long, max_lat, max_long, 1580, 2070)
    displacement_EW = interpolate_matrix(imageEW, rows, cols, min_lat_s2, max_lat_s2, min_long_s2, max_long_s2, lat_vector, long_vector)
    displacement_NS = interpolate_matrix(imageNS, rows, cols, min_lat_s2, max_lat_s2, min_long_s2, max_long_s2, lat_vector, long_vector)

    plot_displacement(displacement_EW,displacement_NS)




if __name__ == '__main__':
    main()