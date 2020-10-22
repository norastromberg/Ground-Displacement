from PIL import Image
import numpy as np
from scipy.interpolate import RectBivariateSpline
import tifffile as tiff

def read_displacement_tif(filepath):
    img = tiff.imread(filepath)
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


def main():
    min_lat = 35.155
    max_lat = 36.14
    min_long = -118.112
    max_long = -116.893
    rows = 10979
    cols = 10979


    image = read_displacement_tif("data/Px1_Num6_DeZoom1_LeChantier.tiff")
    lat_vector, long_vector = create_coordinate_vectors(min_lat, min_long, max_lat, max_long, rows, cols)
    interpolated_matrix = interpolate_matrix(image, rows, cols, min_lat, max_lat, min_long, max_long, lat_vector, long_vector)




if __name__ == '__main__':
    main()