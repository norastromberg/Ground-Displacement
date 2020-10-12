import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


def get_coordinate_range(asc_lat_min, asc_lat_max, asc_long_min,asc_long_max,desc_lat_min,desc_lat_max,desc_long_min,desc_long_max):
    min_lat = max(asc_lat_min, desc_lat_min)
    max_lat = min(asc_lat_max, desc_lat_max)
    min_long = max(asc_long_min, desc_long_min)
    max_long = min(asc_long_max, desc_long_max)

    return min_lat, max_lat, min_long, max_long


def generate_interpolation_function(matrix_to_interpolate, rows, cols, lat_min, lat_max, long_min, long_max):
    x = np.linspace(long_min, long_max, cols)
    y = np.linspace(lat_min, lat_max, rows)
    interp_spline = RectBivariateSpline(y, x, matrix_to_interpolate)

    return interp_spline


def read_image_to_matrix(filepath, rows, cols):
    x = np.fromfile(filepath, dtype=">f4")
    img = np.reshape(x, [rows, cols])

    return img


def interpolate_matrix(min_lat, min_long, max_lat, max_long, rows, cols, interpolation_function):
    x0 = np.linspace(min_long, max_long, cols)
    y0 = np.linspace(min_lat, max_lat, rows)
    interpolated_matrix = interpolation_function(y0, x0)
    fig, ax = plt.subplots()
    ax.matshow(interpolated_matrix, cmap=plt.cm.Blues)
    #plt.show()

    return interpolated_matrix


def program():
    asc_min_lat, asc_max_lat, asc_min_long, asc_max_long = [35.51650381422822, 35.91371799195127, -117.84217825827987, -117.32224343296268]
    desc_min_lat, desc_max_lat, desc_min_long, desc_max_long = [35.514734076106755, 35.91445275684194, -117.84199673504028, -117.32131055881945]
    min_lat, max_lat, min_long, max_long = [35.51650381422822, 35.91371799195127, -117.84199673504028, -117.32224343296268]
    asc_los_displacement = read_image_to_matrix("data/ASC_cropped_Displacement_Geocoded_2806_1007.data/displacement_VV.img", 1611, 2094)
    asc_incidenceangle = read_image_to_matrix("data/ASC_cropped_Displacement_Geocoded_2806_1007.data/incidenceAngleFromEllipsoid.img", 1611, 2094)
    desc_los_displacement = read_image_to_matrix("data/DESC_cropped_Geocoded_Displacement_2206_1607.data/displacement_VV.img", 1609, 2092)
    desc_incidenceangle = read_image_to_matrix("data/DESC_cropped_Geocoded_Displacement_2206_1607.data/incidenceAngleFromEllipsoid.img", 1609, 2092)
    asc_displacement_interpolation_function = generate_interpolation_function(asc_los_displacement, 1611, 2094, asc_min_lat, asc_max_lat, asc_min_long, asc_max_long)
    asc_incidenceangle_interpolation_function = generate_interpolation_function(asc_incidenceangle, 1611, 2094, asc_min_lat, asc_max_lat, asc_min_long, asc_max_long)
    desc_displacement_interpolation_function = generate_interpolation_function(desc_los_displacement, 1609, 2092, desc_min_lat, desc_max_lat, desc_min_long, desc_max_long)
    desc_incidenceangle_interpolation_function = generate_interpolation_function(desc_incidenceangle, 1609, 2092, desc_min_lat, desc_max_lat, desc_min_long, desc_max_long)
    asc_interpolated_los_displacement = interpolate_matrix(min_lat,min_long,max_lat, asc_max_long,1600,2050, asc_displacement_interpolation_function)
    asc_interpolated_incidenceangle = interpolate_matrix(min_lat,min_long,max_lat,max_long, 1600,2050, asc_incidenceangle_interpolation_function)
    desc_interpolated_los_displacement = interpolate_matrix(min_lat,min_long,max_lat,max_long,1600, 2050, desc_displacement_interpolation_function)
    desc_interpolated_incidenceangle = interpolate_matrix(min_lat,min_long,max_lat,max_long,1600, 2050, desc_incidenceangle_interpolation_function)
    plt.plot(asc_interpolated_los_displacement)
    plt.show()
    plt.plot(desc_interpolated_los_displacement)
    plt.show()
    plt.plot(asc_interpolated_incidenceangle)
    plt.show()
    plt.plot(desc_interpolated_incidenceangle)
    plt.show()

def main():
    program()

if __name__ == '__main__':
    main()
