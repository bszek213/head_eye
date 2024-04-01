# Eye and Head analysis: Brian Szekely
import matplotlib.pyplot as plt
import numpy as np
import vedb_gaze
import vedb_store
# from vedb_odometry import vedb_calibration #remove this after odopy is fixed
from odopy import headCalibrate
import os
import yaml
from tqdm import tqdm
from sys import argv
import rigid_body_motion as rbm
# from scipy.spatial.transform import Rotation
from scipy import signal
# from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter
from matplotlib.patches import Ellipse
# import seaborn as sns
from scipy.interpolate import interp1d
import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime
# import pickle
n_cores = None 
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
"""
Current issues
-Cannot use any qmul on the calibrated odo data as the orientation data are just a unit quaternion [1,0,0,0]

-my qmul version and Peter's return different results...

-for this to work: eye_world = rbm.lookup_angular_velocity("eye","world_coord",
                                                as_dataarray=True,
                                                represent_in='eye')
i need to upsample the eye data to the sampling rate of the head which obviously creates noise
gaze_3d_right_list
media/bszekely/BrianDrive/2022_02_09_13_40_13_test_walk_session
-in the test take, the head shakes and nodes happen roughly around 2 min and 38 seconds from session start
-check to see if the az and el are canceled out

"""
def quaternion_multiply_array(q1, q2):
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack((w, x, y, z), axis=-1)

def quaternion_conjugate_array(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    return np.stack((w, -x, -y, -z), axis=-1)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def quaternion_inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / np.linalg.norm(q)#np.array([w, -x, -y, -z])

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_average(q1, q2):
    return normalize_quaternion((q1 + q2) / 2.0)

def quaternion_to_rotation_matrix(q):
    #Convert quaternion to rotation matrix using direction cosine matrix method
    q = normalize_quaternion(q)
    
    #quaternion components
    w, x, y, z = q
    
    #elements of the rotation matrix
    r11 = 1 - 2*y**2 - 2*z**2
    r12 = 2*(x*y + w*z)
    r13 = 2*(x*z - w*y)
    r21 = 2*(x*y - w*z)
    r22 = 1 - 2*x**2 - 2*z**2
    r23 = 2*(y*z + w*x)
    r31 = 2*(x*z + w*y)
    r32 = 2*(y*z - w*x)
    r33 = 1 - 2*x**2 - 2*y**2
    
    #Construct the rotation matrix
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    
    return rotation_matrix

def spherical_to_cartesian(azimuth, elevation):
    #Convert azimuth and elevation to Cartesian coordinates
    # x = np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    x = np.sin(np.radians(azimuth)) / np.cos(np.radians(elevation))
    # y = np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
    y = np.cos(np.radians(azimuth)) / np.cos(np.radians(elevation))
    z = np.sin(np.radians(elevation))
    return x, y, z

def check_for_mp4_files(folder_path):
    mp4_files = [file for file in os.listdir(folder_path) if file.endswith('.mp4')]
    return bool(mp4_files)


def custom_moving_average_wrap_df(df, column, window_size_percent, median=False):
    result_mean = pd.Series(index=df.index, dtype=float)
    result_std = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        window_size = int(len(df) * window_size_percent / 100)
        start_idx = i - window_size
        end_idx = i + window_size + 1

        if start_idx < 0:
            window_data = pd.concat([df[column].iloc[start_idx:], df[column].iloc[:end_idx]])
        elif end_idx > len(df):
            window_data = pd.concat([df[column].iloc[start_idx:], df[column].iloc[:end_idx - len(df)]])
        else:
            window_data = df[column].iloc[start_idx:end_idx]
        if median == False:
            result_mean.iloc[i] = window_data.mean()
            result_std.iloc[i] = window_data.std()
        else:
            result_mean.iloc[i] = window_data.median()
            result_std.iloc[i] = window_data.std()

    return result_mean,result_std

class eyeHead():
    def __init__(self):
        print("instantiate eyeHead class object")
        self.input_folder = argv[1]
        self.fov_horiz = 125 #deg 
        self.fov_vertical = 110 #deg
        self.pupil_fs = 1/120
        #Extrinsics reference frame
        # self.rotation_extrinsics = [0, 0.5235988, 0]
        self.rotation_extrinsics = [0,30,0] #degrees
        self.rotation_extrinsics_quat = [0.9781476007, 0, 0.2079116908, 0]
        self.trans_y = 1.25 #inches
        self.trans_x = 0 #inches
        self.trans_z = 1.15 #inches

    def eye_utils(self):
        self.calibration_epoch = 0  # first only
        self.validation_epoch = 0  # first only
        # Get session files
        self.ses = vedb_store.Session.from_folder(argv[1], raise_error=False, overwrite_user_info=True)
        self.world_time_file, self.world_vid_file = self.ses.paths['world_camera']
        self.eye_left_time_file, self.eye_left_file = self.ses.paths['eye_left']
        self.eye_right_time_file, self.eye_right_file = self.ses.paths['eye_right']

        if os.path.exists(os.path.join(self.input_folder, 'marker_times.yaml')):
            with open(os.path.join(self.input_folder, 'marker_times.yaml'), 'r') as file:
                self.marker_times = yaml.safe_load(file)
        else:
            self.marker_times = vedb_store.utils.specify_marker_epochs(self.input_folder)
        mark_times = self.marker_times
        self.validation_times = os.path.join(self.input_folder,
                                        'marker_times.yaml')
        self.eye_left_st, self.eye_left_end = vedb_store.utils.get_frame_indices(
                        *mark_times["calibration_times"][self.calibration_epoch], self.ses.get_video_time("eye_left"))
        eye_right_st, eye_right_end = vedb_store.utils.get_frame_indices(
                        *mark_times["calibration_times"][self.calibration_epoch], self.ses.get_video_time("eye_right"))

    def calibration_markers_find(self):
        print('Find calibration markers')
        calibration_marker_file = os.path.join(self.input_folder,'calibration_markers.npz')
        if os.path.exists(calibration_marker_file):
            self.calibration_markers = dict(np.load(calibration_marker_file))
        else:    
            self.calibration_markers = vedb_gaze.marker_detection.find_concentric_circles(self.world_vid_file, self.world_time_file, 
                                                                    start_frame=self.marker_times['calibration_frames'][0][0], 
                                                                    end_frame=self.marker_times['calibration_frames'][0][1], 
                                                                    n_cores=n_cores,
                                                                    progress_bar=tqdm)
            temp_calib = self.calibration_markers
            print(f'saving calibration markers to this file path: {calibration_marker_file}')
            np.savez(calibration_marker_file, **temp_calib)
    def load_pylids(self):
        print('LOAD PYLIDS DETECTED PUPILS')
        self.pupil = dict(left=dict(np.load(os.path.join(self.input_folder,'pupil_left.npz'), allow_pickle=True)),
                right=dict(np.load(os.path.join(self.input_folder,'pupil_right.npz'), allow_pickle=True)),
            )
        # with open(os.path.join(self.input_folder,'eye1.pkl'), 'rb') as f:
        #     self.pupil_left = pickle.load(f)
        # with open(os.path.join(self.input_folder,'eye0.pkl'), 'rb') as f:
        #     self.pupil_right = pickle.load(f)
        # print(self.pupil_right)
    def calculate_calibration(self):
        self.all_timestamps = self.ses.get_video_time('world_camera')
        if len(self.calibration_markers['norm_pos']) == 0:
            # Failed
            self.calibration_markers = None
            self.calibration_markers_filtered = None
            self.calibration_input = None
        else:
            self.calibration_markers_filtered = vedb_gaze.marker_parsing.find_epochs(
                self.calibration_markers, self.all_timestamps)
            if len(self.calibration_markers_filtered) == 0:
                self.calibration_markers_filtered = None
                calibration_input = self.calibration_markers
            else:
                self.calibration_input = self.calibration_markers_filtered[self.calibration_epoch]
        print("PUPIL CALIBRATION")
        self.calibration = {}
        self.calibration['left'] = vedb_gaze.calibration.Calibration(self.pupil['left'], 
                                                                        self.calibration_input,
                                                                        self.ses.world_camera.resolution,
                                                                        lambd_list=[1e-06,
                                                                                    2.9286445646252375e-06,
                                                                                    8.576958985908945e-06,
                                                                                    2.5118864315095822e-05,
                                                                                    7.356422544596421e-05,
                                                                                    0.00021544346900318845,
                                                                                    0.000630957344480193,
                                                                                    0.0018478497974222907,
                                                                                    0.0054116952654646375,
                                                                                    0.01584893192461114,
                                                                                    0.04641588833612782,
                                                                                    0.1359356390878527,
                                                                                    0.3981071705534969,
                                                                                    1.165914401179831,
                                                                                    3.414548873833601,
                                                                                    10.0],
                                                                        max_stds_for_outliers=3.0,
                                                                        )
        self.calibration['right'] = vedb_gaze.calibration.Calibration(self.pupil['right'], 
                                                                        self.calibration_input,
                                                                        self.ses.world_camera.resolution,
                                                                        lambd_list=[1e-06,
                                                                                    2.9286445646252375e-06,
                                                                                    8.576958985908945e-06,
                                                                                    2.5118864315095822e-05,
                                                                                    7.356422544596421e-05,
                                                                                    0.00021544346900318845,
                                                                                    0.000630957344480193,
                                                                                    0.0018478497974222907,
                                                                                    0.0054116952654646375,
                                                                                    0.01584893192461114,
                                                                                    0.04641588833612782,
                                                                                    0.1359356390878527,
                                                                                    0.3981071705534969,
                                                                                    1.165914401179831,
                                                                                    3.414548873833601,
                                                                                    10.0],
                                                                        max_stds_for_outliers=3.0,
                                                                        )
    def validation_markers_calc(self):
        print("VALIDATION MARKERS")
        validation_marker_file = os.path.join(self.input_folder,'validation_markers.npz')
        if os.path.exists(validation_marker_file):
            self.validation_markers = dict(np.load(validation_marker_file))
        else:
            self.validation_markers = vedb_gaze.marker_detection.find_checkerboard(self.world_vid_file, self.world_time_file, 
                                                                start_frame=self.marker_times['validation_frames'][self.validation_epoch][0], 
                                                                end_frame=self.marker_times['validation_frames'][self.validation_epoch][1],
                                                                n_cores=n_cores,
                                                                progress_bar=tqdm)
            temp = self.validation_markers
            print(f'saving validation markers to this file path: {validation_marker_file}')
            np.savez(validation_marker_file, **temp)
    def gaze_calc(self):
        print('CALCULATE GAZE')
        self.gaze = {}
        for lr in ['left', 'right']:
            if self.calibration[lr] is None:
                self.gaze[lr] = None
                print(f'no gaze for {lr}')
            else:
                self.gaze[lr] = self.calibration[lr].map(self.pupil[lr])
        return self.gaze
    def gaze_error(self):
        # Check for failed validation detection & filter validation points for spurious detections
        if len(self.validation_markers['norm_pos']) == 0:
            # Failed
            self.validation_markers = None
            self.validation_markers_filtered = None
        else:
            self.validation_markers_filtered = vedb_gaze.marker_parsing.find_epochs( 
                self.validation_markers, self.all_timestamps, 
                aspect_ratio_threshold=None, # Don't do aspect ratio thresholding for validation
                )
            if len(self.validation_markers_filtered) == 0:
                self.validation_markers_filtered = None
                self.error_input = [self.validation_markers]
            else:
                self.error_input = self.validation_markers_filtered
        print('ERROR CALCULATION')
        self.error = {}
        for lr in ['left', 'right']:
            if self.calibration[lr] is None:
                self.error[lr] = None
            else:
                try:
                    if self.validation_markers_filtered is None:
                        self.error[lr] = [vedb_gaze.error_computation.compute_error(ei,
                                                                            self.gaze[lr],
                                                                            image_resolution=self.ses.world_camera.resolution[::-1],
                                                                            method='tps',
                                                                            lambd=1.0,
                                                                            cluster_reduce_fn=None,)
                                    for ei in self.error_input]
                    else:
                        self.error[lr] = [vedb_gaze.error_computation.compute_error(ei,
                                                                            self.gaze[lr],
                                                                            image_resolution=self.ses.world_camera.resolution[::-1],)
                                    for ei in self.error_input]
                except:
                    print("%s eye error calculation failed."%lr)
                    self.error[lr] = None 
        # print(self.gaze)
        left_error = np.mean(self.error['left'][0]['gaze_err_angle'])
        right_error = np.mean(self.error['right'][0]['gaze_err_angle'])
        print(f'Gaze left error: {left_error} degrees') 
        print(f'Gaze right error: {right_error} degrees') 
        # print(self.error)
        # #TODO: ADD error plot - h_im = ax.imshow(err['gaze_err_image'], **im_kw) current error due to gaze error image being nans
        # # if np.mean(self.pupil['left']['confidence']) > np.mean(self.pupil['right']['confidence']):
        # #     keep_confidence = self.pupil['left']['confidence']
        # # else:
        # #     keep_confidence = self.pupil['right']['confidence']
        # # self.gaze['confidence'] = keep_confidence
        # #plot the error
        # vedb_gaze.visualization.plot_error(self.error['right'],self.gaze['right'])
        # plt.show()
    def qc_plot(self):
        # QC Visualization
        fpath = os.path.join(self.input_folder,'qc_plot.png')
        pipeline_outputs = dict(pupil=self.pupil,
                                calibration_marker_all=self.calibration_markers,
                                calibration_marker_filtered=self.calibration_markers_filtered if self.calibration_markers_filtered is None else self.calibration_markers_filtered[
                                    self.calibration_epoch],
                                validation_marker_all=self.validation_markers,
                                validation_marker_filtered=self.validation_markers_filtered if self.validation_markers_filtered is None else self.validation_markers_filtered,
                                calibration=self.calibration,
                                gaze=self.gaze,
                                error=self.error,)
        vedb_gaze.visualization.plot_session_qc(self.ses,
                                                **pipeline_outputs,
                                                fpath=fpath,
                                                )
        plt.close()
    def head_calibration(self):
        odo = headCalibrate.headCalibrate()#vedb_calibration.vedbCalibration()
        odo.set_odometry_local(self.input_folder)
        odo.start_end_plot()
        odo.t265_to_head_trans()
        self.calib_odo = odo.get_calibrated_odo()
        self.rbm_head_eye = odo.get_rbm()
        self.odometry = odo.get_odometry()
        # odo.calc_head_orientation()
        # head_roll, head_pitch, head_yaw = odo.get_head_orientation()
        # print(head_pitch.linear_acceleration)
        # plt.plot(head_pitch.linear_acceleration)
        # plt.show()
        # odo.plot()
    def convert_norm_pos_to_degrees(self):
        """
        Right now I am overwriting the previous values. 
        in the future I need to save these to a new variable
        convert norm pos[0 - 1] into the FOV of the pupil cameras
        """
        self.gaze['left']['gaze_degrees'] = np.zeros((len(self.gaze['left']['norm_pos'][:,0]), 2))
        self.gaze['right']['gaze_degrees'] = np.zeros((len(self.gaze['right']['norm_pos'][:,0]), 2))
        self.gaze['left']['gaze_degrees'][:,0] = self.gaze['left']['norm_pos'][:,0] * self.fov_horiz
        self.gaze['right']['gaze_degrees'][:,0] = self.gaze['right']['norm_pos'][:,0] * self.fov_horiz
        self.gaze['left']['gaze_degrees'][:,1] = self.gaze['left']['norm_pos'][:,1] * self.fov_vertical
        self.gaze['right']['gaze_degrees'][:,1] = self.gaze['right']['norm_pos'][:,1] * self.fov_vertical
        #orient about 0
        horizontal_FOV =  self.fov_horiz / 2
        vertical_FOV = self.fov_vertical / 2
        self.gaze['left']['gaze_degrees'][:,0] = self.gaze['left']['gaze_degrees'][:,0] - horizontal_FOV 
        self.gaze['left']['gaze_degrees'][:,1] = self.gaze['left']['gaze_degrees'][:,1] - vertical_FOV
        self.gaze['right']['gaze_degrees'][:,0] = self.gaze['right']['gaze_degrees'][:,0] - horizontal_FOV 
        self.gaze['right']['gaze_degrees'][:,1] = self.gaze['right']['gaze_degrees'][:,1] - vertical_FOV

    def set_equal_timestamps(self):
        # smoothed_azimuth = savgol_filter(self.gaze['right']['gaze_degrees'][:,0], 
        #                                  window_length=int((1/self.pupil_fs) * 0.42),
        #                                  polyorder=2)
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.gaze['right']['gaze_degrees'][:,0], label='Original Azimuth Velocity', color='blue', linewidth=3.5)
        # plt.plot(smoothed_azimuth, label='Smoothed Azimuth Velocity', color='red', linestyle='--')
        # plt.xlabel('Time')
        # plt.ylabel('Velocity (deg/s)')
        # plt.title('Savitzky-Golay Filter')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        original_signal = self.gaze['left']['gaze_degrees'][:, 1]
        # Specify the window size for the median filter
        initial_window_size = int((1 / self.pupil_fs) * 0.75)
        window_size = initial_window_size if initial_window_size % 2 == 1 else initial_window_size + 1
        
        # Apply the median filter
        self.gaze['right']['gaze_degrees'][:, 0] = medfilt(self.gaze['right']['gaze_degrees'][:, 0], kernel_size=window_size)
        self.gaze['right']['gaze_degrees'][:, 1] = medfilt(self.gaze['right']['gaze_degrees'][:, 1], kernel_size=window_size)
        self.gaze['left']['gaze_degrees'][:, 0] = medfilt(self.gaze['left']['gaze_degrees'][:, 0], kernel_size=window_size)
        self.gaze['left']['gaze_degrees'][:, 1] = medfilt(self.gaze['left']['gaze_degrees'][:, 1], kernel_size=window_size)
        smoothed_signal = self.gaze['left']['gaze_degrees'][:, 1]
        # Plot original and filtered signals
        # plt.figure(figsize=(10, 6))
        # plt.plot(original_signal, label='Original Signal', color='blue', linewidth=3.5)
        plt.plot(smoothed_signal, label='Smoothed Signal', color='red', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.title('Median Filter')
        plt.legend()
        plt.grid(True)
        plt.show()

    def setup_ref_frame(self):
        #World Reference Frame
        # rbm.register_frame("world", update=True)
        #Head Reference Frame
        # rbm.register_frame("head",
        #                     translation=self.calib_odo.lin_pos,
        #                     rotation=self.calib_odo.ang_pos,
        #                     timestamps=self.calib_odo.time,
        #                     parent="world",
        #                     update=True,
        #                     )
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.gaze['left']['gaze_degrees'][:,1])
        # self.set_equal_timestamps()
        #find the closest timestamps in a the longer eye gaze data structure
        #the gaze left and gaze right data are not equal in length
        if len(self.gaze['left']['timestamp']) > len(self.gaze['right']['timestamp']):
            longer_list = self.gaze['left']['timestamp']
            shorter_list = self.gaze['right']['timestamp']
        else:
            longer_list = self.gaze['right']['timestamp']
            shorter_list = self.gaze['left']['timestamp']

        closest_values = []
        indices = []
        print('Find timestamps in both arrays to make them equivalent')
        for value in tqdm(shorter_list):
            closest_value = longer_list[np.abs(longer_list - value).argmin()]
            closest_values.append(closest_value)
            closest_index = np.where(longer_list == closest_value)[0][0]
            indices.append(closest_index)
        eye_global_timestamps = np.array(closest_values)
        #TODO: HIGH PRIORITY - MAKE THE eye_global_timestamps AS DATETIME64 NUMPY TO MATCH PETERS RBM PACKAGE
        #CURRENTLY DOESNT WORK AS THEY DO NOT HAVE THE SAME DATE OR TIME AS THE FLOATS FROM GAZE ARE NOT IN REFERENCE
        #TO UTC

        #Linearly interpolate missing values
        nan_indices = np.isnan(eye_global_timestamps)
        non_nan_indices = ~nan_indices
        #Interpolate only NaN values
        eye_global_timestamps_interpolated = np.copy(eye_global_timestamps)
        eye_global_timestamps_interpolated[nan_indices] = np.interp(
            np.arange(len(eye_global_timestamps))[nan_indices],
            non_nan_indices,
            eye_global_timestamps[non_nan_indices]
        )
        time_start = self.calib_odo.time.values[0]
        differences = np.diff(eye_global_timestamps_interpolated)
        mean_difference = np.mean(differences[differences != 0]) 
        save_list = [time_start] 
        for i in differences:
            if i == 0:#if the time diff is zero use the mean difference between time values
                i = mean_difference
            time_start = np.datetime64(time_start + np.timedelta64(int(i), 's'))
            save_list.append(time_start)
        eye_global_timestamps_datetime = np.array(save_list, dtype='datetime64[ns]')

        #Convert the interpolated array to datetime64
        # eye_global_timestamps_datetime = np.array(eye_global_timestamps_interpolated, dtype='datetime64[ns]')

        if len(self.gaze['left']['timestamp']) > len(self.gaze['right']['timestamp']):
            gaze_left_equal = self.gaze['left']['gaze_degrees'][indices,:]
            gaze_right_equal = self.gaze['right']['gaze_degrees']
        else:
            gaze_left_equal = self.gaze['left']['gaze_degrees']
            gaze_right_equal = self.gaze['right']['gaze_degrees'][indices,:]

        #Create "3D" gaze matrix with the z axis (torsion) with 0
        self.gaze_3d_left = np.array((gaze_left_equal[:,0].reshape(-1), 
                       gaze_left_equal[:,1].reshape(-1),
                       np.zeros((len(gaze_left_equal[:,0])))
                       )).T
        self.gaze_3d_right = np.array((gaze_right_equal[:,0].reshape(-1), 
                       gaze_right_equal[:,1].reshape(-1),
                       np.zeros((len(gaze_right_equal[:,0])))
                       )).T
        


        # self.gaze_3d_left = np.array((self.gaze['left']['gaze_degrees'][:,0].reshape(-1), 
        #                self.gaze['left']['gaze_degrees'][:,1].reshape(-1),
        #                np.zeros((len(self.gaze['left']['gaze_degrees'][:,0])))
        #                )).T
        # self.gaze_3d_right = np.array((self.gaze['right']['gaze_degrees'][:,0].reshape(-1), 
        #                self.gaze['right']['gaze_degrees'][:,1].reshape(-1),
        #                np.zeros((len(self.gaze['right']['gaze_degrees'][:,0])))
        #                )).T

        # put the eye data into the world coordinate system representation
        # flip azimuth and torsion
        # self.gaze_3d_left[:, [0, -1]] = self.gaze_3d_left[:, [-1, 0]]
        # self.gaze_3d_right[:, [0, -1]] = self.gaze_3d_right[:, [-1, 0]]
        gaze_3d_left_list,gaze_3d_right_list = [], []
        for val in self.gaze_3d_left:
            gaze_3d_left_list.append(spherical_to_cartesian(val[0],val[1]))

        for val in self.gaze_3d_right:
            gaze_3d_right_list.append(spherical_to_cartesian(val[0],val[1]))

        #mulitply the eye in world camera by the extrinsic rotation
        eye_orient_left = rbm.shortest_arc_rotation(np.array(gaze_3d_left_list),np.array((1, 0, 0)))
        eye_orient_right = rbm.shortest_arc_rotation(np.array(gaze_3d_right_list),np.array((1, 0, 0)))

        # eye_orient_left[:, [1, -1]] = eye_orient_left[:, [-1, 1]]
        # eye_orient_right[:, [1, -1]] = eye_orient_right[:, [-1, 1]]

        #create cycloplean eye
        eye_orient_left_normalized = normalize_quaternion(eye_orient_left)
        eye_orient_right_normalized = normalize_quaternion(eye_orient_right)
        combined_eye_orientation = quaternion_average(eye_orient_left_normalized, eye_orient_right_normalized)

        # eye_in_head_coordinates = combined_eye_orientation
        extrinsic_quat = rbm.from_euler_angles(roll=self.rotation_extrinsics[0],
                              pitch=self.rotation_extrinsics[1],
                              yaw=self.rotation_extrinsics[2])
        eye_in_head_coordinates = rbm.qmul(combined_eye_orientation, extrinsic_quat)

        # eye_in_head_coordinates = quaternion_multiply(extrinsic_quat, 
        #                                 quaternion_multiply(combined_eye_orientation, 
        #                                 quaternion_conjugate(extrinsic_quat)))

        #apply this v′=q⋅v⋅q−1
        #upsample to remove velocity error
        odo_ang_vel_vestib = rbm.transform_vectors(self.odometry.angular_velocity,outof='t265_world',into='t265_vestibular')
        if np.isnan(odo_ang_vel_vestib.values).any():
            nan_indices = np.isnan(odo_ang_vel_vestib.values)
            x = np.arange(odo_ang_vel_vestib.values.shape[0])
            interpolated_data = np.empty_like(odo_ang_vel_vestib.values)
            for i in range(odo_ang_vel_vestib.values.shape[1]):
                valid_indices = ~nan_indices[:, i]
                interp_func = interp1d(x[valid_indices], odo_ang_vel_vestib.values[valid_indices, i], kind='linear', fill_value='extrapolate')
                interpolated_data[:, i] = interp_func(x)
        else:
            interpolated_data = odo_ang_vel_vestib.values

        # odo_ang_vel_vestib = self.calib_odo.ang_vel.values
        ang_vel_t265_down = signal.resample(interpolated_data, len(eye_in_head_coordinates)) #len(self.calib_odo.ang_pos.values)
        
        #Perform quaternion multiplication for each angular velocity vector separately
        num_samples = ang_vel_t265_down.shape[0]
        num_axes = 3  #Assuming the angular velocity contains x, y, and z components
        transformed_angular_velocity = np.zeros((num_samples, num_axes))

        # eye_in_head_coordinates = rbm.qinv(eye_in_head_coordinates)
        for i in range(num_samples):
            #Get the quaternion representing the rotation for the current sample
            quaternion = eye_in_head_coordinates[i]

            #Convert the current angular velocity vector to a quaternion with zero scalar part
            # print(f'norm before: {np.linalg.norm(ang_vel_t265_down[i])}')
            # print(ang_vel_t265_down[i])
            rotated_ang_vel = np.dot(quaternion_to_rotation_matrix(quaternion),ang_vel_t265_down[i])
            # ang_vel_quaternion = np.concatenate(([0], ang_vel_t265_down[i]))
            # # print(ang_vel_quaternion)
            # #Perform quaternion multiplication to rotate the angular velocity quaternion
            # rotated_ang_vel_quaternion = quaternion_multiply(quaternion,
            #                                                 quaternion_multiply(ang_vel_quaternion, 
            #                                                                     quaternion_inverse(quaternion)))
            # print(f'norm after: {np.linalg.norm(rotated_ang_vel)}')
            # print(rotated_ang_vel)
            # Extract the vector part from the rotated angular velocity quaternion
            transformed_angular_velocity[i] = rotated_ang_vel
        self.transformed_angular_velocity_world = transformed_angular_velocity
        #plot hist of azimuth and elevation
        # fig, ax = plt.subplots(nrows=2,ncols=1)
        # # plt.plot(transformed_angular_velocity[:,0],label='transformed x')
        # ax[0].hist(transformed_angular_velocity[:,1],label='transformed y',bins=500)
        # ax[1].hist(transformed_angular_velocity[:,2],label='transformed z',bins=500)
        # ax[0].hist(ang_vel_t265_down[:,1],label='untransformed y',bins=500,alpha=0.4)
        # ax[1].hist(ang_vel_t265_down[:,2],label='untransformed z',bins=500,alpha=0.4)
        # plt.ylabel('odo ang vel (rad/s)')
        # plt.xlabel('sample')
        # ax[0].legend()
        # ax[1].legend()
        # plt.show()


        # fig, ax = plt.subplots(1,2)
        # plt.figure()
        # # Convert quaternion array to Euler angle array
        # euler_array = np.zeros((eye_orient_left.shape[0], 3))
        # for i, quaternion in enumerate(eye_orient_left):
        #     modified_quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # Change order to [w, x, y, z]
        #     r = Rotation.from_quat(modified_quaternion)
        #     euler = r.as_euler('xyz',degrees=True)
        #     euler_array[i] = euler

        # plt.plot(eye_global_timestamps,euler_array[:,1],label='gaze x')
        # ax[1].plot(self.gaze['left']['timestamp'],eye_orient[:,1],label='gaze quaternion')
        #transform extrinsics to quaternions

        ### CANNOT DO ANY OF THIS AS THE CALIBRATED ODO DATA ARE JUST A UNIT QUATERNION
        #Apply extrinsics rotation to eye data: DO I APPLY THE QINV HERE?
        # head_ang_pos = signal.resample(self.calib_odo.ang_pos.values, len(eye_in_head_coordinates))
        # eye_result = quaternion_multiply(head_ang_pos, 
        #                                                quaternion_multiply(eye_in_head_coordinates, 
        #                                                quaternion_conjugate(head_ang_pos)))
        # plt.figure()
        # plt.plot(head_ang_pos[:, 1],label='head x quat component') #self.calib_odo.time.values, 
        # plt.plot(eye_in_head_coordinates[:, 1],label='eye in head x quat component')
        # plt.plot(eye_result[:, 1],label='eye in world x quat component')
        # plt.tight_layout()
        # plt.legend()
        # plt.title('my qmul way')
        # plt.show()
        # plt.figure()
        # plt.plot(eye_orient_right_normalized)
        # plt.show()
        #Setup eye transform
        # print(rbm.render_tree("world"))

        # self.rbm_head_eye.register_frame("eye",
        #                     rotation=eye_in_head_coordinates,
        #                     timestamps=self.calib_odo.time.values,#eye_global_timestamps_datetime
        #                     parent="t265_calib",
        #                     update=True,
        #                     )
        # print(self.odometry.angular_velocity)
        # eye_world = rbm.transform_vectors(self.odometry.angular_velocity,
        #                        outof="t265_world", into="eye")
        
        # eye_world = rbm.lookup_angular_velocity("eye","world_coord",
        #                                         as_dataarray=True,
        #                                         represent_in='world_coord')
        # eye_world = np.rad2deg(eye_world)
        
        #Plot eye_world on the first y-axis

        # self.gaze['left']['degrees'].values[:, 1] 
        # self.gaze['left']['timestamps']

        # plt.figure()
        # plt.plot(self.calib_odo['ang_vel'],label='head')
        # plt.plot(eye_world.values[:, 1],label='eye world') #self.calib_odo.time.values, 
        # plt.show()
        # # plt.close()
        # plt.rcParams.update({
        #     'font.size': 16,
        #     'axes.labelweight': 'bold',
        #     'axes.titlesize': 16,
        #     'legend.fontsize': 14,
        # })
        # plt.rcParams["font.weight"] = "bold"

        # plt.figure(figsize=(8, 6))
        # # plt.hist(eye_world.values[:, 0],bins=1000,color='r')
        # # plt.hist(eye_world.values[:, 1],bins=1000,color='b',alpha=0.4)
        # plt.hist2d(x=eye_world.values[:, 0], y=eye_world.values[:, 1], bins=7500, cmap='hot')
        # plt.colorbar(label='Count')
        # plt.title('Heatmap of eye_world')
        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.xlim([-50,50])
        # plt.ylim([-50,50])
        # plt.show()

        # fig, ax1 = plt.subplots()
        # ax1.plot(self.calib_odo.time.values, eye_world[:, 0],
        #           label='Azimuth Velocity (deg/s)', color='blue',linewidth=3.5)
        # ax1.plot(self.calib_odo.time.values, eye_world[:, 1], 
        #          label='Elevation Velocity (deg/s)', color='green',linewidth=3.5)
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('Eye Velocity (deg/s)', color='black')
        # ax1.legend(loc='upper left')

        # # Create a second y-axis
        # ax2 = ax1.twinx()
        # ax2.plot(self.calib_odo.time.values, self.calib_odo.lin_vel[:,2].values,
        #           color='red',linewidth=3.5)
        # ax2.set_ylabel('Linear Vertical Head Velocity (m/s)', color='black')
        # ax2.legend(loc='upper right')
        # plt.show()
    
    def heat_map(self):
        print(f'data before nan removal: {self.transformed_angular_velocity_world.shape}')
        print(self.transformed_angular_velocity_world)
        #remove nans
        cleaned_data = self.transformed_angular_velocity_world[~np.isnan(
            self.transformed_angular_velocity_world).any(axis=1)]  
        print(f'data after nan removal: {cleaned_data.shape}')  
        #mean and standard deviation
        means = np.mean(cleaned_data, axis=0)
        stds = np.std(cleaned_data, axis=0)

        #filter data within 2 standard deviations from the mean
        filtered_data = []
        print(f'data before outlier removal: {cleaned_data.shape}')
        for row in cleaned_data:
            if ((abs(row[0]) <= means[0] + 3 * stds[0]) and 
                (abs(row[1]) <= means[1] + 3 * stds[1]) and 
                (abs(row[2]) <= means[2] + 3 * stds[2])):
                filtered_data.append(row)
            # else:
            #     filtered_data.append(np.full_like(row, np.nan))  # Replace the entire row with NaNs

        transformed_angular_velocity_world_filtered = np.array(filtered_data)
        print(f'data after outlier removal: {transformed_angular_velocity_world_filtered.shape}')

        #2D hist
        data_first_axis = np.rad2deg(transformed_angular_velocity_world_filtered[:, 2])
        data_last_axis = np.rad2deg(transformed_angular_velocity_world_filtered[:, 1])
        plt.figure(figsize=(10, 10))
        plt.hist2d(data_first_axis, data_last_axis, bins=850, cmap='viridis')
        plt.xlabel('Eye Yaw Velocity (deg/s)')
        plt.ylabel('Eye Pitch Velocity (deg/s)')

        #cov matrix
        cov_matrix = np.cov(data_first_axis, data_last_axis)
        #eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        #angle of rotation from the eigenvectors
        angle = np.degrees(np.arctan2(*eigenvectors[:, 1]))
        #confidence ellipse
        confidence_level = 0.95
        width = 2 * np.sqrt(eigenvalues[0] * np.abs(np.log(1 - confidence_level)))
        height = 2 * np.sqrt(eigenvalues[1] * np.abs(np.log(1 - confidence_level)))
        ellipse = Ellipse(xy=(np.mean(data_first_axis), np.mean(data_last_axis)), width=width, height=height,
                        angle=angle, edgecolor='white', facecolor='none', linestyle='--', linewidth=2)
        if height < width:
            lims = height
        else:
            lims = width
        plt.xlim([-lims , lims])
        plt.ylim([-lims , lims])
        plt.gca().add_patch(ellipse)
        plt.tight_layout()
        plt.savefig(os.path.join(self.input_folder,'eye_ang_vel_world.png'),dpi=400)
        plt.close()

    def gaze_step_cycle(self):
        #brian_walk_test: walk seg: 20:00:10 - 20:01:20
        #2022_02_09_13_40_13_test_walk_session walk seg:  21:44:20 - 21:56:00
        # plt.figure()
        # plt.plot(self.odometry.time.values,self.odometry.linear_velocity.values[:,1])
        # plt.show()
        # plt.close()

        #some takes have nans in the odometry data
        if np.isnan(self.odometry.linear_velocity.values).any():
            nan_indices = np.isnan(self.odometry.linear_velocity.values)
            x = np.arange(self.odometry.linear_velocity.values.shape[0])
            interpolated_data = np.empty_like(self.odometry.linear_velocity.values)
            for i in range(self.odometry.linear_velocity.values.shape[1]):
                valid_indices = ~nan_indices[:, i]
                interp_func = interp1d(x[valid_indices], self.odometry.linear_velocity.values[valid_indices, i], kind='linear', fill_value='extrapolate')
                interpolated_data[:, i] = interp_func(x)
        else:
            interpolated_data = self.odometry.linear_velocity.values
        odo_pos_down = signal.resample(interpolated_data, 
                                            len(self.transformed_angular_velocity_world))
        
        #downsample time
        time_series = pd.to_datetime(self.odometry.time.values)
        df = pd.DataFrame(index=time_series, data={'values': range(len(time_series))})
        downsample_factor = len(time_series) / len(self.transformed_angular_velocity_world)
        downsampled_indices = np.floor(np.arange(0, len(time_series), downsample_factor)).astype(int)
        resampled_df = df.iloc[downsampled_indices].interpolate(method='time')
        resampled_time_index = resampled_df.index
        time_odo_eye = resampled_time_index.to_numpy()

        #extract walking segment
        time_odo_eye = pd.to_datetime(time_odo_eye)
        target_time_start = "21:44:20"#input("Enter the first target time (format: HH:MM:SS): ")
        target_time_end = "21:56:00"#input("Enter the second target time (format: HH:MM:SS): ")

        target1 = datetime.strptime(target_time_start, '%H:%M:%S')
        target2 = datetime.strptime(target_time_end, '%H:%M:%S')

        target_date = pd.to_datetime(time_odo_eye[0]).date()
        target1 = target1.replace(year=target_date.year, month=target_date.month, day=target_date.day)
        target2 = target2.replace(year=target_date.year, month=target_date.month, day=target_date.day)

        time_diff1 = np.abs(time_odo_eye - np.datetime64(target1))
        time_diff2 = np.abs(time_odo_eye - np.datetime64(target2))
        closest_index1 = np.argmin(time_diff1)
        closest_index2 = np.argmin(time_diff2)

        walk_times = time_odo_eye[closest_index1:closest_index2]
        odo_pos_down_walk = odo_pos_down[closest_index1:closest_index2,:]
        eye_vel_walk = self.transformed_angular_velocity_world[closest_index1:closest_index2,:]

        find_peaks_step, _= find_peaks(odo_pos_down_walk[:,1],distance=45)

        # plt.figure()
        # plt.plot(walk_times,odo_pos_down_walk[:,1])
        # plt.scatter(walk_times[find_peaks_step],odo_pos_down_walk[find_peaks_step,1],marker='*',color='red')
        # plt.show()

        list_gait, list_yaw, list_pitch = [], [], []
        for i in tqdm(range(len(find_peaks_step)-1)):
            diff_samples = find_peaks_step[i+1] - find_peaks_step[i]
            gait_percentage = np.linspace(0,100,num=diff_samples)
            eye_step = np.rad2deg(eye_vel_walk[find_peaks_step[i]:find_peaks_step[i+1],:])
            list_gait.append(gait_percentage)
            list_yaw.append(eye_step[:,2])
            list_pitch.append(eye_step[:,1])
    
        list_gait = np.concatenate(list_gait)
        list_yaw = np.concatenate(list_yaw)
        list_pitch = np.concatenate(list_pitch)

        df = pd.DataFrame({'gait_per': list_gait, "yaw_vel": list_yaw, "pitch_vel": list_pitch})
        df.sort_values(by='gait_per',inplace=True)
        df['norm'] = np.linalg.norm(df[['yaw_vel', 'pitch_vel']].values, axis=1)

        # df['yaw_vel'] = savgol_filter(df['yaw_vel'],231,2)
        # df['pitch_vel'] = savgol_filter(df['pitch_vel'],231,2)
        # df['norm'] = savgol_filter(df['pitch_vel'],231,2)
        
        mean_yaw_vel = df['yaw_vel'].mean()
        std_yaw_vel = df['yaw_vel'].std()

        mean_pitch_vel = df['pitch_vel'].mean()
        std_pitch_vel = df['pitch_vel'].std()

        df_filtered = df[~((df['yaw_vel'] > mean_yaw_vel + 3 * std_yaw_vel) | (df['yaw_vel'] < mean_yaw_vel - 3 * std_yaw_vel) |
                        (df['pitch_vel'] > mean_pitch_vel + 3 * std_pitch_vel) | (df['pitch_vel'] < mean_pitch_vel - 3 * std_pitch_vel))]

        yaw_vel_roll, yaw_std_vel = custom_moving_average_wrap_df(df_filtered,'yaw_vel',5)
        pitch_vel_roll, pitch_std_vel = custom_moving_average_wrap_df(df_filtered,'pitch_vel',5)
        norm_vel_roll, norm_std_vel = custom_moving_average_wrap_df(df_filtered,'norm',5)

        norm_std_vel = savgol_filter(norm_std_vel,27,2)
    
        # fig, ax = plt.subplots(nrows=3,ncols=1)
        # ax[0].plot(df['gait_per'],yaw_vel_roll,linewidth=3)
        # ax[1].plot(df['gait_per'],pitch_vel_roll,linewidth=3)
        plt.figure(figsize=(8,8))
        plt.plot(df_filtered['gait_per'], norm_vel_roll, linewidth=3, label='Rolling Mean')
        plt.fill_between(df_filtered['gait_per'], norm_vel_roll - norm_std_vel, 
                         norm_vel_roll + norm_std_vel, alpha=0.5, label='Rolling STDEV')
        plt.xlabel('Step Cycle (%)')
        plt.ylabel('Eye Velocity Norm (deg/s)')
        plt.tight_layout()
        # plt.legend()
        plt.savefig(os.path.join(self.input_folder,'eye_ang_vel_world_step_cycle.png'),dpi=400)
        plt.close()


        # plt.figure()
        # plt.plot(walk_times[find_peaks_stride],odo_pos_down_walk[find_peaks_stride,1],marker='*')
        # plt.plot(walk_times,odo_pos_down_walk[:,1])
        # plt.show()



    def run_analysis(self):
        #vedb takes with video
        self.eye_utils()
        self.calibration_markers_find()
        self.load_pylids()
        self.calculate_calibration()
        self.validation_markers_calc()
        gaze = self.gaze_calc()
        self.gaze_error()
        self.qc_plot()
        self.convert_norm_pos_to_degrees()
        self.head_calibration()
        #RBM
        self.setup_ref_frame()
        self.heat_map()
        self.gaze_step_cycle()

def main():
    eyeHead().run_analysis()
if __name__ == "__main__":
    main()