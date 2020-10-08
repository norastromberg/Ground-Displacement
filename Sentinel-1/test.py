import imageio
import numpy as np
import numpy as np


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

def read_image_size():
    asc_file = open("data/asc_coordinate_info.txt")
    desc_file = open("data/desc_coordinate_info.txt")
    asc_size = asc_file.read().split(",")
    desc_size = desc_file.read().split(",")

    return asc_size,desc_size

def program():
    asc_size, desc_size = read_image_size()

def main():
    #img1 = read_file('/Users/norabrask/PycharmProjects/ProjectAssignment/Ground-Displacement/Sentinel-1/sample.hdr')
    #img2 = read_file('/Users/norabrask/PycharmProjects/ProjectAssignment/Ground-Displacement/Sentinel-1/sample.hdr')
    #compute_deformation_east_west(img1, img2, 10, 12, 11, 13, 0.5, 0.5)

    # calculate_dv_de(1,1,90,45,45,45,1,1)
    #print(read_image_to_matrix("/Users/Sigrid/Documents/prosjektoppgave/Ground-Displacement/Sentinel-1/data/ASC_cropped_Displacement_Geocoded_2806_1007.data/localIncidenceAngle.img",2094,1611))
    #read_dim("Sentinel-1/data/ASC_cropped_Displacement_Geocoded_2806_1007.data/ASC_cropped_Displacement_Geocoded_2806_1007.dim")
    program()


if __name__ == '__main__':
    main()
