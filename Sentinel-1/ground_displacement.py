import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


def read_image_to_matrix(filepath, rows, cols):
    x = np.fromfile(filepath, dtype=">f4")
    img = np.reshape(x, [rows, cols])

    return img


def get_coordinate_range(asc_lat_min, asc_lat_max, asc_long_min, asc_long_max, desc_lat_min, desc_lat_max, desc_long_min, desc_long_max):
    min_lat = max(asc_lat_min, desc_lat_min)
    max_lat = min(asc_lat_max, desc_lat_max)
    min_long = max(asc_long_min, desc_long_min)
    max_long = min(asc_long_max, desc_long_max)

    return min_lat, max_lat, min_long, max_long


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


def calculate_dv_de(los1, los2, inc1, inc2, az1, az2):
    A = np.array([[np.cos(inc1), -np.cos(az1) * np.sin(inc1)], [np.cos(inc2), -np.cos(az2) * np.sin(inc2)]])
    B = np.array([[los1], [los2]])
    X = np.linalg.inv(A).dot(B)
    dv = X[0][0]
    de = X[1][0]

    return dv, de


def calculate_de_dv_matrix(lat_vector, long_vector, asc_disp_matrix, asc_inc_matrix, desc_disp_matrix, desc_inc_matrix, asc_az, desc_az):
    dv_matrix = np.zeros((len(lat_vector), long_vector.shape[0]))
    de_matrix = np.zeros((len(lat_vector), long_vector.shape[0]))
    for i in range(len(lat_vector)):
        for j in range(len(long_vector)):
            asc_los = asc_disp_matrix[i, j]
            asc_inc = asc_inc_matrix[i, j]
            desc_los = desc_disp_matrix[i, j]
            desc_inc = desc_inc_matrix[i, j]
            dv, de = calculate_dv_de(asc_los, desc_los, asc_inc, desc_inc, asc_az, desc_az)
            dv_matrix[i][j] = dv
            de_matrix[i][j] = de

    return dv_matrix, de_matrix


def plot_displacement(dv_matrix, de_matrix):
    fig1, ax1 = plt.subplots()
    caxis1 = ax1.matshow(dv_matrix, cmap=plt.get_cmap("RdBu"))
    fig1.colorbar(caxis1)
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.5165", "35.6158", "35.7151", "35.8144", "35.9137"])
    ax1.set(xlabel="Longitude", ylabel="Latitude")
    ax1.set_title("Displacement vertical")

    fig2, ax2 = plt.subplots()
    caxis2 = ax2.matshow(de_matrix, cmap=plt.get_cmap("RdBu"))
    fig2.colorbar(caxis2)
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.5165", "35.6158", "35.7151", "35.8144", "35.9137"])
    ax2.set(xlabel="Longitude", ylabel="Latitude")
    ax2.set_title("Displacement East-West")

    plt.show()


def program():
    az_asc = -13.07251464606799
    az_desc = -166.9596229763622
    asc_min_lat, asc_max_lat, asc_min_long, asc_max_long = [35.51650381422822, 35.91371799195127, -117.84217825827987, -117.32224343296268]
    desc_min_lat, desc_max_lat, desc_min_long, desc_max_long = [35.514734076106755, 35.91445275684194, -117.84199673504028, -117.32131055881945]
    min_lat, max_lat, min_long, max_long = [35.51650381422822, 35.91371799195127, -117.84199673504028, -117.32224343296268]

    asc_los_displacement = read_image_to_matrix("data/ASC_cropped_Displacement_Geocoded_2806_1007_new.data/displacement_VV.img", 1583, 2073)
    asc_incidenceangle = read_image_to_matrix("data/ASC_cropped_Displacement_Geocoded_2806_1007_new.data/incidenceAngleFromEllipsoid.img", 1583, 2073)
    desc_los_displacement = read_image_to_matrix("data/DESC_cropped_Geocoded_Displacement_2206_1607_new.data/displacement_VV.img", 1581, 2092)
    desc_incidenceangle = read_image_to_matrix("data/DESC_cropped_Geocoded_Displacement_2206_1607_new.data/incidenceAngleFromEllipsoid.img", 1581, 2092)

    lat_vector, long_vector = create_coordinate_vectors(min_lat, min_long, max_lat, max_long, 1580, 2070)

    asc_interpolated_los_displacement_matrix = interpolate_matrix(asc_los_displacement, 1583, 2073, asc_min_lat, asc_max_lat, asc_min_long, asc_max_long, lat_vector, long_vector)
    asc_interpolated_incidence_matrix = interpolate_matrix(asc_incidenceangle, 1583, 2073, asc_min_lat, asc_max_lat, asc_min_long, asc_max_long, lat_vector, long_vector)

    desc_interpolated_los_displacement_matrix = interpolate_matrix(desc_los_displacement, 1581, 2092, desc_min_lat, desc_max_lat, desc_min_long, desc_max_long, lat_vector, long_vector)
    desc_interpolated_incidence_matrix = interpolate_matrix(desc_incidenceangle, 1581, 2092, desc_min_lat, desc_max_lat, desc_min_long, desc_max_long, lat_vector, long_vector)

    dv_matrix, de_matrix = calculate_de_dv_matrix(lat_vector, long_vector, asc_interpolated_los_displacement_matrix,
                                                  asc_interpolated_incidence_matrix, desc_interpolated_los_displacement_matrix, desc_interpolated_incidence_matrix, az_asc, az_desc)
    plot_displacement(dv_matrix, de_matrix)

def main():
    program()

if __name__ == '__main__':
    main()
