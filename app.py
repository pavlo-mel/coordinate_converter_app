import streamlit as st
import pandas as pd

from object_gps_position import compute_K, compute_camera_rotation, compute_object_world_coords, compute_object_gps_coords


st.markdown(
    """
    <h1 style="text-align: center;">
        <span style="color: wheat;">Object Coordinate Converter:</span><br>
        <span style="color: skyblue;">From Detection to GPS</span>
    </h1>
    """,
    unsafe_allow_html=True
)
    
st.subheader("Image Resolution")
col5, col6 = st.columns(2)
with col5:
    W = st.number_input("Image width (pixels)", value=3840)
with col6:
    H = st.number_input("Image height (pixels)", value=2160)

st.subheader("Object Pixel Coordinates (Detection)")
col7, col8 = st.columns(2)
with col7:
    x_im = st.number_input("Pixel x-coordinate", value=2000)
with col8:
    y_im = st.number_input("Pixel y-coordinate", value=1000)


st.markdown("""
    <style>
    .stRadio > div {
        display: flex;
        flex-direction: row;
    }
    </style>
    """, unsafe_allow_html=True)
st.subheader("Select Camera Type")
camera_type = st.radio(
    "Choose the camera configuration:",
    ("Wide-Angle Camera", "Telephoto Camera", "Other")
)

# Set default values based on the selected camera type
if camera_type == "Wide-Angle Camera":
    f_x_default, f_y_default = 24, 24
    sensor_width_default, sensor_height_default = 17.3, 13.0
elif camera_type == "Telephoto Camera":
    f_x_default, f_y_default = 162, 162
    sensor_width_default, sensor_height_default = 6.4, 4.8
else:
    f_x_default, f_y_default = 50, 50
    sensor_width_default, sensor_height_default = 15, 15
    

st.subheader("Camera Parameters")
col1, col2 = st.columns(2)
with col1:
    f_x = st.number_input("Focal length (f_x) in mm", value=f_x_default)
    sensor_width = st.number_input("Sensor width (mm)", value=sensor_width_default)
with col2:
    f_y = st.number_input("Focal length (f_y) in mm", value=f_y_default)
    sensor_height = st.number_input("Sensor height (mm)", value=sensor_height_default)

sensor_size = (sensor_width, sensor_height)

# Group camera setup fields logically using columns
st.subheader("Camera Setup")
col3, col4 = st.columns(2)
with col3:
    camera_altitude = st.number_input("Camera altitude (m)", value=3.0)
    alpha = st.number_input("Camera tilt (degrees)", value=45.0)
with col4:
    lat0 = st.number_input("Camera GPS: latitude", value=37.7749)
    lon0 = st.number_input("Camera GPS: longitude", value=-122.4194)

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4F8EAD;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #3a6b8f;
    }
    .stButton {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# button to run the computation
if st.button("Compute object GPS coordinates"):
    # STEP 1: Compute camera intrinsics and the tilt matrix
    K = compute_K((f_x, f_y), sensor_size, (W, H))
    R_tilt = compute_camera_rotation(alpha)
    
    # STEP 2: Compute the object coordinates in the world
    object_world_coords = compute_object_world_coords((x_im, y_im), K, R_tilt, camera_altitude)
    
    # STEP 3: Compute the GPS coordinates of the object
    object_lat, object_lon = compute_object_gps_coords(object_world_coords, (lat0, lon0))
    
    # display the results
    st.success(f"Object GPS Coordinates: Latitude = {object_lat}, Longitude = {object_lon}")
    
    # option to save the result as a CSV file
    coords_data = pd.DataFrame({
        'Latitude': [object_lat],
        'Longitude': [object_lon]
    })
    
    csv = coords_data.to_csv(index=False)
    
    st.download_button(
        label="Download Coordinates as CSV",
        data=csv,
        file_name='object_coordinates.csv',
        mime='text/csv'
    )

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        © 2024 Pavlo Melnyk. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)