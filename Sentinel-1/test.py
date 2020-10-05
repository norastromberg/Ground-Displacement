import sys
import imageio
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


def read_file(filepath):
    img = imageio.imread(filepath, 'hdr')
    # print(img.shape)
    # print(type(img))
    img_nparray = np.asarray(img)
    # print('np shape',img_nparray.shape, img.ndim)
    # print(img_nparray.shape[0])
    return img_nparray

#Forutsetter at begge bildene har samme dimensjoner:
def compute_deformation_east_west(img1, img2, inc1, inc2, az1, az2, wavelength1, wavelength2):
    east_west_matrix = np.zeros((img1.shape[0], img1.shape[1]), list)
    for i in range(img1.shape[0]):
        print(i)
        for j in range(img1.shape[1]):
            ph_LOS1 = img1[i, j][0]
            ph_LOS2 = img2[i, j][0]

            dv, de = calculate_dv_de(ph_LOS1, ph_LOS2, inc1, inc2, az1, az2, wavelength1, wavelength2)

            east_west_matrix[i][j] = np.asarray([dv,de])

    # print(east_west_matrix)
    return(east_west_matrix)



def calculate_dv_de(ph_LOS1, ph_LOS2, inc1, inc2, az1, az2, wavelength1, wavelength2):
    pi = np.pi
    #Regn ut dv og de gitt at
    A = np.array([[np.cos(inc1), -np.cos(az1)*np.sin(inc1)], [np.cos(inc2), -np.cos(az2)*np.sin(inc2)]])
    print(A)
    B = np.array([[ph_LOS1*(wavelength1/(4*pi))],[ph_LOS2*(wavelength1/(4*pi))]])
    print(B)
    X = np.linalg.inv(A).dot(B)
    dv = X[0][0]
    de = X[1][0]
    print('dv = ', dv)
    print('de = ', de)
    return dv, de



def main():
    img1 = read_file('/Users/norabrask/PycharmProjects/ProjectAssignment/Ground-Displacement/Sentinel-1/sample.hdr')
    img2 = read_file('/Users/norabrask/PycharmProjects/ProjectAssignment/Ground-Displacement/Sentinel-1/sample.hdr')
    compute_deformation_east_west(img1, img2, 10, 12, 11, 13, 0.5, 0.5)

    #calculate_dv_de(1,1,90,45,45,45,1,1)

if __name__ == '__main__':
    main()