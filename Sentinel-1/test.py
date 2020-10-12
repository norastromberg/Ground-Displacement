import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as cKD


def read_file(filepath):
    img = imageio.imread(filepath, 'img')
    img_nparray = np.asarray(img)

    return img_nparray


def read_img_files():
    lat = imageio.imread('/Users/Sigrid/Documents/prosjektoppgave/Ground-Displacement/Sentinel-1/data/ASC_cropped_Displacement_Geocoded_2806_1007.data/latitude.img')
    lon = imageio.imread('/Users/Sigrid/Documents/prosjektoppgave/Ground-Displacement/Sentinel-1/data/ASC_cropped_Displacement_Geocoded_2806_1007.data/longitude.img')
    print('latitude')
    print(lat)
    print('longitude')
    print(lon)

# Forutsetter at begge bildene har samme dimensjoner:
def compute_deformation_east_west(img1, img2, inc1, inc2, az1, az2):
    east_west_matrix = np.zeros((img1.shape[0], img1.shape[1]), list)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            ph_LOS1 = img1[i, j][0]
            ph_LOS2 = img2[i, j][0]
            dv, de = calculate_dv_de(ph_LOS1, ph_LOS2, inc1, inc2, az1, az2)
            east_west_matrix[i][j] = np.asarray([dv, de])

    return east_west_matrix


def calculate_dv_de(los1, los2, inc1, inc2, az1, az2):
    A = np.array([[np.cos(inc1), -np.cos(az1) * np.sin(inc1)], [np.cos(inc2), -np.cos(az2) * np.sin(inc2)]])
    B = np.array([[los1], [los2]])
    X = np.linalg.inv(A).dot(B)
    dv = X[0][0]
    de = X[1][0]
    print('dv = ', dv)
    print('de = ', de)

    return dv, de

def read_image_to_matrix(filepath, rows, cols):
    data_type = np.dtype('<f4')
    x = np.fromfile(filepath, dtype=data_type)
    img = np.reshape(x, [rows, cols])
    return img

def generate_dataframe(filepath):
    df = pd.read_table(filepath, skiprows=6)
    return df

def plot():
    df_asc = generate_dataframe("data/ASC_cropped_Displacement_Geocoded_2806_1007.data/ASC_INFO.txt")
    df_desc = generate_dataframe("data/DESC_cropped_Geocoded_Displacement_2206_1607.data/DESC_INFO.txt")
    asc_lat = df_asc["Latitude"].head(10000)
    asc_long = df_asc["Longitude"].head(10000)
    desc_lat = df_asc["Latitude"].head(10000)
    desc_long = df_asc["Longitude"].head(10000)
    plt.plot(asc_lat,asc_long,"r--",desc_lat,desc_long,"bs")
    plt.show()

def find_min_max_coordinates(df_asc,df_desc):
    asc_lat_min = df_asc["Latitude"].min()
    desc_lat_min = df_desc["Latitude"].min()
    asc_lat_max = df_asc["Latitude"].max()
    desc_lat_max = df_desc["Latitude"].max()
    asc_long_min = df_asc["Longitude"].min()
    desc_long_min = df_desc["Longitude"].min()
    asc_long_max = df_asc["Longitude"].max()
    desc_long_max = df_desc["Longitude"].max()
    min_lat = max(asc_lat_min, desc_lat_min)
    max_lat = min(asc_lat_max, desc_lat_max)
    min_long = max(asc_long_min, desc_long_min)
    max_long = min(asc_long_max, desc_long_max)
    print("ASC: " , asc_lat_min,asc_lat_max,asc_long_min,asc_long_max)
    print("DESC: " , desc_lat_min, desc_lat_max, desc_long_min, desc_long_max)
    return min_lat, max_lat, min_long, max_long

def generate_coordinate_grid(df_asc, df_desc):
    min_lat, max_lat, min_long, max_long = find_min_max_coordinates(df_asc, df_desc)
    x = np.linspace(min_long, max_long,2050)
    y = np.linspace(min_lat, max_lat, 1600)
    coordinate_matrix = np.zeros([1600, 2050], dtype=object)
    for i in range(1600):
        for j in range(2050):
            array = [y[i], x[j]]
            coordinate_matrix[i][j] = array

    return coordinate_matrix


def build_cKDtrees(df_asc, df_desc):
    asc_lat_longs = generate_lat_longs(df_asc)
    desc_lat_longs = generate_lat_longs(df_desc)
    asc_tree = cKD(asc_lat_longs)
    desc_tree = cKD(desc_lat_longs)

    return asc_tree, desc_tree


def generate_lat_longs(df):
    lat_longs = df[["Latitude", "Longitude"]]

    return lat_longs.values.tolist()


def find_nearest_points_and_info(lat_long,df_asc, df_desc):
    asc_tree, desc_tree = build_cKDtrees(df_asc, df_desc)
    nearest_asc_point = asc_tree.query([lat_long], k=1)
    nearest_desc_point = desc_tree.query([lat_long], k=1)
    index_asc = nearest_asc_point[1][0]
    index_desc = nearest_desc_point[1][0]
    info_asc = df_asc[["displacement_VV", "incidenceAngleFromEllipsoid"]].iloc[index_asc]
    info_desc = df_desc[["displacement_VV", "incidenceAngleFromEllipsoid"]].iloc[index_desc]

    return info_asc, info_desc


def calculate_dv_de(lat, long, df_asc, df_desc):
    info_asc, info_desc = find_nearest_points_and_info([lat, long], df_asc, df_desc)
    inc_asc = info_asc["incidenceAngleFromEllipsoid"]
    inc_desc = info_desc["incidenceAngleFromEllipsoid"]
    disp_asc = info_asc["displacement_VV"]
    disp_desc = info_desc["displacement_VV"]
    az_asc = -13.07251464606799
    az_desc = -166.9596229763622

    A = np.array([[np.cos(inc_asc), -np.cos(az_asc) * np.sin(inc_asc)], [np.cos(inc_desc), -np.cos(az_desc) * np.sin(inc_desc)]])
    B = np.array([[disp_asc], [disp_desc]])
    X = np.linalg.inv(A).dot(B)
    dv = X[0][0]
    de = X[1][0]
    print('dv = ', dv)
    print('de = ', de)

    return dv,de


def generate_displacement_matrix(df_asc, df_desc):
    coordinate_matrix = generate_coordinate_grid(df_asc, df_desc)
    for row in coordinate_matrix:
        print(row)
        for col in coordinate_matrix:
            lat = coordinate_matrix[row][col][0]
            long = coordinate_matrix[row][col][1]
            dv,de = calculate_dv_de(lat, long, df_asc, df_desc)
            coordinate_matrix[row][col] = [dv,de]

    return coordinate_matrix


def program():
    df_asc = generate_dataframe("data/ASC_cropped_Displacement_Geocoded_2806_1007.data/ASC_INFO.txt")
    df_desc = generate_dataframe("data/DESC_cropped_Geocoded_Displacement_2206_1607.data/DESC_INFO.txt")
    print(find_min_max_coordinates(df_asc,df_desc))

    #displacement_matrix = generate_displacement_matrix(df_asc,df_desc)
    #file = open("data/matrix.txt", "w")
    #file.write(displacement_matrix)
    #file.close()
    #return displacement_matrix



def main():
    program()

if __name__ == '__main__':
    main()
