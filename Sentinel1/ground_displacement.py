import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


def read_image_to_matrix(filepath, rows, cols):
    x = np.fromfile(filepath, dtype=">f4")
    img = np.reshape(x, [rows, cols])

    return img


def create_coordinate_vectors(min_lat, min_long, max_lat, max_long, rows, cols):
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


def decompose_displacement(asc_los, desc_los, asc_inc, desc_inc, asc_az, desc_az, north_zero):
    if (north_zero):
        A = np.array([[np.cos(asc_inc), -np.cos(asc_az) * np.sin(asc_inc)],
                      [np.cos(desc_inc), -np.cos(desc_az) * np.sin(desc_inc)]])
    else:
        A = np.array([[np.cos(asc_inc), np.sin(asc_az) * np.sin(asc_inc)],
                      [np.cos(desc_inc), np.sin(desc_az) * np.sin(desc_inc)]])
    B = np.array([[asc_los], [desc_los]])
    X = np.linalg.inv(A).dot(B)
    dv = X[0][0]
    dh = X[1][0]

    return dh, dv


def create_decomposed_matrix(rows, cols, asc_disp_matrix, asc_inc_matrix, desc_disp_matrix, desc_inc_matrix, asc_az,
                             desc_az, north_zero):
    dv_matrix = np.zeros([rows, cols])
    dh_matrix = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            asc_los = asc_disp_matrix[i, j]
            asc_inc = asc_inc_matrix[i, j]
            desc_los = desc_disp_matrix[i, j]
            desc_inc = desc_inc_matrix[i, j]
            if (north_zero):
                dh, dv = decompose_displacement(asc_los, desc_los, asc_inc, desc_inc, asc_az, desc_az, north_zero)
            else:
                dh, dv = decompose_displacement(asc_los, desc_los, asc_inc, desc_inc, asc_az, desc_az, north_zero)
            dh_matrix[i][j] = dh
            dv_matrix[i][j] = dv

    return dh_matrix, dv_matrix


def plot_displacement(rows, cols, matrix1, matrix2, vmin1, vmax1, vmin2, vmax2, title1, title2, scale):
    fig1, ax1 = plt.subplots()
    caxis1 = ax1.matshow(matrix1, cmap=plt.get_cmap("RdBu_r"), vmin=vmin1, vmax=vmax1)
    clb1 = fig1.colorbar(caxis1)
    clb1.set_label('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(1, cols, 5), ["-117.84", "-117.71", "-117.58", "-117.45", "-117.32"])
    plt.yticks(np.linspace(1, rows, 5), ["35.91", "35.81", "35.71", "35.62", "35.52"])
    ax1.set(xlabel="Longitude", ylabel="Latitude")
    ax1.set_title(title1, weight="bold")
    ax1.xaxis.set_ticks_position('bottom')
    scalebar = ScaleBar(scale, location="lower left")
    plt.gca().add_artist(scalebar)

    fig2, ax2 = plt.subplots()
    caxis2 = ax2.matshow(matrix2, cmap=plt.get_cmap("RdBu_r"), vmin=vmin2, vmax=vmax2)
    clb2 = fig2.colorbar(caxis2)
    clb2.set_label('[m]', rotation=90, weight="bold")
    plt.xticks(np.linspace(1, cols, 5), ["-117.84", "-117.71", "-117.58", "-117.45", "-117.32"])
    plt.yticks(np.linspace(1, rows, 5), ["35.91", "35.81", "35.71", "35.62", "35.52"])
    ax2.set(xlabel="Longitude", ylabel="Latitude")
    ax2.set_title(title2, weight="bold")
    ax2.xaxis.set_ticks_position('bottom')
    scalebar = ScaleBar(scale, location="lower left")
    plt.gca().add_artist(scalebar)


def get_matrices_for_least_squares_s1():
    rows = 1581
    cols = 2068
    asc_az = -13.07251464606799
    desc_az = -166.9596229763622
    min_lat, max_lat, min_long, max_long = [35.517, 35.913, -117.841, -117.323]

    asc_displacement = read_image_to_matrix("Sentinel1/data/ASC/displacement_VV.img", rows, cols)
    asc_incidenceangle = read_image_to_matrix("Sentinel1/data/ASC/incidenceAngleFromEllipsoid.img", rows, cols)

    desc_displacement = read_image_to_matrix("Sentinel1/data/DESC/displacement_VV.img", rows, cols)
    desc_incidenceangle = read_image_to_matrix("Sentinel1/data/DESC/incidenceAngleFromEllipsoid.img", rows, cols)

    return rows, cols, asc_az, desc_az, min_lat, max_lat, min_long, max_long, asc_displacement, asc_incidenceangle, desc_displacement, desc_incidenceangle

    # asc_los_displacement = read_image_to_matrix(
    #     "Sentinel1/data/ASC_cropped_Displacement_Geocoded_2806_1007_new.data/displacement_VV.img", 1583, 2073)
    # asc_incidenceangle = read_image_to_matrix(
    #     "Sentinel1/data/ASC_cropped_Displacement_Geocoded_2806_1007_new.data/incidenceAngleFromEllipsoid.img", 1583, 2073)
    # desc_los_displacement = read_image_to_matrix(
    #     "Sentinel1/data/DESC_cropped_Geocoded_Displacement_2206_1607_new.data/displacement_VV.img", 1581, 2092)
    # desc_incidenceangle = read_image_to_matrix(
    #     "Sentinel1/data/DESC_cropped_Geocoded_Displacement_2206_1607_new.data/incidenceAngleFromEllipsoid.img", 1581, 2092)
    #
    # lat_vector, long_vector = create_coordinate_vectors(min_lat, min_long, max_lat, max_long, 1580, 2070)
    #
    # asc_interpolated_los_displacement_matrix = interpolate_matrix(asc_los_displacement, 1583, 2073, asc_min_lat,
    #                                                               asc_max_lat, asc_min_long, asc_max_long, lat_vector,
    #                                                               long_vector)
    # asc_interpolated_incidence_matrix = interpolate_matrix(asc_incidenceangle, 1583, 2073, asc_min_lat, asc_max_lat,
    #                                                        asc_min_long, asc_max_long, lat_vector, long_vector)
    #
    # desc_interpolated_los_displacement_matrix = interpolate_matrix(desc_los_displacement, 1581, 2092, desc_min_lat,
    #                                                                desc_max_lat, desc_min_long, desc_max_long,
    #                                                                lat_vector, long_vector)
    # desc_interpolated_incidence_matrix = interpolate_matrix(desc_incidenceangle, 1581, 2092, desc_min_lat, desc_max_lat,
    #                                                         desc_min_long, desc_max_long, lat_vector, long_vector)
    # return asc_interpolated_los_displacement_matrix, asc_interpolated_incidence_matrix, desc_interpolated_los_displacement_matrix, desc_interpolated_incidence_matrix,az_asc,az_desc


def program():
    rows = 1581
    cols = 2068
    asc_az = -13.07251464606799
    desc_az = -166.9596229763622

    asc_displacement_matrix = read_image_to_matrix("data/ASC/displacement_VV.img", rows, cols)
    asc_incidence_matrix = read_image_to_matrix("data/ASC/incidenceAngleFromEllipsoid.img", rows, cols)
    desc_displacement_matrix = read_image_to_matrix("data/DESC/displacement_VV.img", rows, cols)
    desc_incidence_matrix = read_image_to_matrix("data/DESC/incidenceAngleFromEllipsoid.img", rows, cols)

    de_matrix, dv_matrix_north_zero = create_decomposed_matrix(rows, cols, asc_displacement_matrix,
                                                               asc_incidence_matrix, desc_displacement_matrix,
                                                               desc_incidence_matrix, asc_az, desc_az, True)
    dn_matrix, dv_matrix_east_zero = create_decomposed_matrix(rows, cols, asc_displacement_matrix,
                                                              asc_incidence_matrix, desc_displacement_matrix,
                                                              desc_incidence_matrix, asc_az, desc_az, False)

    plot_displacement(rows, cols, asc_displacement_matrix, desc_displacement_matrix, -0.8, 0.8, -0.8, 0.8,
                      "LOS displacement (Ascending)", "LOS displacement (Descending)", 20)
    plot_displacement(rows, cols, de_matrix, dv_matrix_north_zero, -1.25, 1.25, -1, 1, "East-West displacement",
                      "Up-Down displacement", 20)
    plot_displacement(rows, cols, dn_matrix, dv_matrix_north_zero, -1.25, 1.25, -1, 1, "North-South displacement",
                      "Up-Down displacement", 20)
    plt.show()


def main():
    program()


if __name__ == '__main__':
    main()
