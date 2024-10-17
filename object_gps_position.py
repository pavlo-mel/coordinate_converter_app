import numpy as np

def compute_K(focal_length, sensor_size, image_size):
    '''
    Computes the camera intrinsics,
    K = [[f_x  0   c_x]
         [0   f_y  c_y]
         [0    0    1 ]],
    where f_x and f_y are the focal lengths in pixels for the x and y directions,
    and c_x and c_y is the principal point coordinates (in the image).
    
    :param focal_length: tuple (f_x, f_y), the focal length of the camera (mm)
    :param sensor_size: tuple (width, height), the sensor size (mm)
    :param image_size: tuple (width, height), the image size in pixels
    
    :return: a 3x3 np.array K
    '''
    # f_x is the focal length in pixels along the x-axis
    f_x = (focal_length[0] / sensor_size[0]) * image_size[0]    
    f_y = (focal_length[1] / sensor_size[1]) * image_size[1]

    # assume the principal point is in the middle of the image:
    c_x = image_size[0] // 2
    c_y = image_size[1] // 2

    K = np.array([
        [f_x, 0,  c_x],
        [0,  f_y, c_y],
        [0,   0,   1]
    ])
    
    return K


def compute_camera_rotation(alpha):
    '''
    Computes the rotation of the camera, R_tilt, that aligns the z-axis with the optical axis
    (assuming the camera is initially pointing downwards, i.e., the bird's eye view).
    
    :param alpha: float, the angle (degrees) between the optical axis and the vertical axis

    :return: a 3x3 np.array R_tilt
    '''
    alpha_rad = np.deg2rad(alpha)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    
    R_tilt = np.array([
        [cos_alpha,  0, sin_alpha],
        [0,          1,         0],
        [-sin_alpha, 0, cos_alpha]
    ])

    return R_tilt


def compute_object_world_coords(object_image_coords, K, R_tilt, camera_altitude):
    '''
    Computes the object coordinates in 3D ([x, y, z]), where the camera position is the origin. 

    Explanation:
    Using the following expression based on projective geometry:
                [x_im, y_im, 1] ~ P [x, y, z, 1], 
    where P = K R_tilt [I 0] (3x4) with I being a 3x3 identity matrix 
    and K the camera intrinsics matrix, we obtain:
                [x_im, y_im, 1] ~ K R_tilt [x, y, z],
                R_tilt^{-1} K^{-1} [x_im, y_im, 1] ~ [x, y, z].

    Since we assumed the camera position is the origin and that the object lies on the ground,
    the last coordinated z is the altitude, i.e., z = camera_altitude.
    Thus, after computing 
                b := R_tilt.T K^{-1} [x_im, y_im, 1], 
    we normalize B s.t. b[-1] == camera_altitude, and find the world coordinates of the object:
                b := camera_altitude * b / b[-1] = [x, y, camera_altitude],

                
    :param object_image_coords: tuple (x_im, y_im), object x and y coordinates in the image
    :param K: np.array, a 3x3 matrix with camera intrinsics
    :param R_tilt: np.array, a 3x3 matrix aligning the z-axis with the optical axis
    :param camera_altitude: float, the distance from the camera to the ground
    
    :return: a np.array representing the 3D coordinates of the object
    '''
    b = R_tilt.T @ np.linalg.inv(K) @ np.array(list(object_image_coords) + [1.])
    b = camera_altitude * b / b[-1]

    return b


def compute_object_gps_coords(object_world_coords, camera_gps):
    '''
    Computes the GPS coordinates of the object given its 3D coordinates in the world CS 
    and the camera GPS coordinates.

    :param object_world coords: np.array, a 3D vector containing the coordinates of the object
    :param camera_gps: tuple, the GPS coordinates of the camera

    :return: a tuple representing the GPS coordinates of the object
    '''
    # since the camera is the origin, the ground distance between the object and the camera 
    # is simply the length of the (x, y) coordinate-vector of the object:
    ground_dist = np.sqrt( np.sum(object_world_coords[:2]**2) ) 
    
    lat0, lon0 = camera_gps
    x_obj, y_obj, _ = object_world_coords
    
    lat1_rad = np.radians(lat0)
    lon1_rad = np.radians(lon0)
  
    R = 6371000.0 # Earth radius in meters

    # compute the distance and bearing from the object coordinates
    bearing_rad = np.arctan2(y_obj, x_obj)  # bearing angle in radians

    # compute the new latitude in radians
    lat2_rad = np.arcsin(np.sin(lat1_rad) * np.cos(ground_dist / R) +
                         np.cos(lat1_rad) * np.sin(ground_dist / R) * np.cos(bearing_rad))

    # compute the new longitude in radians
    lon2_rad = lon1_rad + np.arctan2(np.sin(bearing_rad) * np.sin(ground_dist / R) * np.cos(lat1_rad),
                                     np.cos(ground_dist / R) - np.sin(lat1_rad) * np.sin(lat2_rad))
    
    lat1 = np.degrees(lat1_rad)
    lon1 = np.degrees(lon1_rad)

    return lat1, lon1

    

def main():
    # for example:
    # focal length
    f_x, f_y = 24, 24
    sensor_size = (17.3, 13.) # sensor (width, height) in mm
    # or:
    # f_x, f_y = 162, 162
    # sensor_size = (6.4, 4.8)

    # camera setup:
    camera_altitude = 3.0  # altitude in meters
    alpha = 45  # camera tilt in degrees (between the optical axis and the vertical axis)
    # the GPS coordinates of the camera
    lat0 = 37.7749  # latitude
    lon0 = -122.4194  # longitude

    # image resolution:
    W = 3840  # image width in pixels
    H = 2160  # image height in pixels
    
    # pixel coordinates of the detected object
    x_im = 2000  # Pixel x-coordinate
    y_im = 1000  # Pixel y-coordinate


    # STEP 1: compute camera intrinsics and the tilt matrix
    K = compute_K((f_x, f_y), sensor_size, (W, H))
    R_tilt = compute_camera_rotation(alpha)

    print(f"\nCamera intrinsics:\n{K}\n")
    print(f"\nCamera rotation:\n{R_tilt}")
    # print(R_tilt@R_tilt.T, np.linalg.det(R_tilt))


    # STEP 2: compute the object coordinates in the world
    object_world_coords = compute_object_world_coords((x_im, y_im), K, R_tilt, camera_altitude)

    print(f"\nObject world coords: {object_world_coords}")

    # STEP 3: compute the GPS coordinates of the object
    object_lat, object_lon = compute_object_gps_coords(object_world_coords, (lat0, lon0))
    print(f"\nObject GPS Coordinates: Latitude = {object_lat:.3f}, Longitude = {object_lon:.3f}")


if __name__ == "__main__":
    main()
