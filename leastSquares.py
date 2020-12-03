import numpy as np
import matplotlib.pyplot as plt
from Sentinel1.ground_displacement import get_matrices_for_least_squares_s1
from Sentinel2.ground_displacement_S2 import get_matrices_for_least_squares_s2
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.interpolate import RectBivariateSpline


def fetch_data():
    s1_rows, s1_cols, asc_az, desc_az, s1_min_lat, s1_max_lat, s1_min_long, s1_max_long, \
    asc_displacement, asc_incidenceangle, desc_displacement, desc_incidenceangle = get_matrices_for_least_squares_s1()

    s2_rows, s2_cols, s2_min_lat, s2_max_lat, s2_min_long, s2_max_long, \
    EW_displacement, NS_displacement = get_matrices_for_least_squares_s2()

    return s1_rows, s1_cols, asc_az, desc_az, s1_min_lat, s1_max_lat, s1_min_long, s1_max_long, asc_displacement, \
           asc_incidenceangle, desc_displacement, desc_incidenceangle, s2_rows, \
           s2_cols, s2_min_lat, s2_max_lat, s2_min_long, s2_max_long, EW_displacement, NS_displacement


def create_coordinate_vectors(min_lat, max_lat, min_long, max_long, rows, cols):
    long_vector = np.linspace(min_long, max_long, cols)
    lat_vector = np.linspace(min_lat, max_lat, rows)

    return lat_vector, long_vector


def interpolate_matrix(matrix_to_interpolate, rows, cols, lat_min, lat_max, long_min, long_max, lat_vector,
                       long_vector):
    x = np.linspace(long_min, long_max, cols)
    y = np.linspace(lat_min, lat_max, rows)
    interp_spline = RectBivariateSpline(y, x, matrix_to_interpolate)
    interpolated_matrix = interp_spline(lat_vector, long_vector)

    return interpolated_matrix


def least_squares():
    s1_rows, s1_cols, asc_az, desc_az, s1_min_lat, s1_max_lat, s1_min_long, s1_max_long, asc_displacement, \
    asc_incidenceangle, desc_displacement, desc_incidenceangle, s2_rows, \
    s2_cols, s2_min_lat, s2_max_lat, s2_min_long, s2_max_long, EW_displacement, NS_displacement = fetch_data()

    lat_vector, long_vector = create_coordinate_vectors(s2_min_lat, s2_max_lat, s2_min_long, s2_max_long, s2_rows, s2_cols)
    asc_interpolated_displacement = interpolate_matrix(asc_displacement, s1_rows, s1_cols, s1_min_lat, s1_max_lat,
                                                       s1_min_long, s1_max_long, lat_vector, long_vector)
    asc_interpolated_incidenceangle = interpolate_matrix(asc_incidenceangle, s1_rows, s1_cols, s1_min_lat, s1_max_lat,
                                                         s1_min_long, s1_max_long, lat_vector, long_vector)
    desc_interpolated_displacement = interpolate_matrix(desc_displacement, s1_rows, s1_cols, s1_min_lat, s1_max_lat,
                                                        s1_min_long, s1_max_long, lat_vector, long_vector)
    desc_interpolated_incidenceangle = interpolate_matrix(desc_incidenceangle, s1_rows, s1_cols, s1_min_lat, s1_max_lat,
                                                          s1_min_long, s1_max_long, lat_vector, long_vector)
    dv_matrix = np.zeros((s2_rows, s2_cols))
    de_matrix = np.zeros((s2_rows, s2_cols))
    dn_matrix = np.zeros((s2_rows, s2_cols))
    for i in range(s2_rows - 1):
        print(i)
        for j in range(s2_cols - 1):
            asc_los = asc_interpolated_displacement[i, j]
            asc_inc = asc_interpolated_incidenceangle[i, j]
            desc_los = desc_interpolated_displacement[i, j]
            desc_inc = desc_interpolated_incidenceangle[i, j]
            ew = EW_displacement[i, j]
            ns = NS_displacement[i, j]
            dv, de, dn = calculate_dv_de_dn(asc_los, asc_inc, asc_az, desc_los, desc_inc, desc_az, ew, ns)
            dv_matrix[i][j] = dv
            de_matrix[i][j] = de
            dn_matrix[i][j] = dn

    return dv_matrix, de_matrix, dn_matrix, EW_displacement, NS_displacement


def calculate_dv_de_dn(asc_los, asc_inc, asc_az, desc_los, desc_inc, desc_az, ew, ns):
    b = [asc_los, desc_los, ew, ns]
    A = [[np.cos(asc_inc), -np.cos(asc_az) * np.sin(asc_inc), np.sin(asc_az) * np.sin(asc_inc)],
         [np.cos(desc_inc), -np.cos(desc_az) * np.sin(desc_inc), np.sin(desc_az) * np.sin(desc_inc)],
         [0, 1, 0],
         [0, 0, 1]]
    X = np.linalg.lstsq(A, b)
    dv = X[0][0]
    de = X[0][1]
    dn = X[0][2]
    return dv, de, dn


def plot_displacement(V, EW, NS):
    fig1, ax1 = plt.subplots()
    caxis1 = ax1.matshow(V, cmap=plt.get_cmap("RdBu_r"))
    clb1 = fig1.colorbar(caxis1)
    clb1.set_label('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.9137", "35.8144", "35.7151", "35.6158", "35.5165"])
    ax1.xaxis.set_ticks_position('bottom')  # the rest is the same
    ax1.set(xlabel="Longitude", ylabel="Latitude")
    ax1.set_title("Displacement Vertical", weight="bold")
    scalebar = ScaleBar(10)  # 1 pixel = 10 meter
    plt.gca().add_artist(scalebar)

    fig2, ax2 = plt.subplots()
    caxis2 = ax2.matshow(EW, cmap=plt.get_cmap("RdBu_r"))
    clb2 = fig2.colorbar(caxis2)
    clb2.set_label('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.9137", "35.8144", "35.7151", "35.6158", "35.5165"])
    ax2.xaxis.set_ticks_position('bottom')  # the rest is the same
    ax2.set(xlabel="Longitude", ylabel="Latitude")
    ax2.set_title("Displacement East-West", weight="bold")
    scalebar = ScaleBar(10)  # 1 pixel = 10 meter
    plt.gca().add_artist(scalebar)

    fig3, ax3 = plt.subplots()
    caxis3 = ax3.matshow(NS, cmap=plt.get_cmap("RdBu_r"))
    clb3 = fig3.colorbar(caxis3)
    clb3.set_label('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(0, 2070, 5), ["-117.8422", "-117.7122", "-117.5822", "-117.4522", "-117.3222"])
    plt.yticks(np.linspace(0, 1580, 5), ["35.9137", "35.8144", "35.7151", "35.6158", "35.5165"])
    ax3.xaxis.set_ticks_position('bottom')  # the rest is the same
    ax3.set(xlabel="Longitude", ylabel="Latitude")
    ax3.set_title("Displacement North-South", weight="bold")
    scalebar = ScaleBar(10)  # 1 pixel = 10 meter
    plt.gca().add_artist(scalebar)

    plt.show()


def main():
    dv_matrix, de_matrix, dn_matrix, ew, ns = least_squares()
    plot_displacement(dv_matrix, de_matrix, dn_matrix)


if __name__ == '__main__':
    main()
