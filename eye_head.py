# Eye and Head analyss: Brian Szekely
import matplotlib.pyplot as plt
import numpy as np
import vedb_gaze
import vedb_store
from vedb_odometry import vedb_calibration
import os
import yaml
from tqdm import tqdm
from sys import argv
import pickle
n_cores = 8 #check your hardware specs

def eye_utils():
    calibration_epoch = 0  # first only
    validation_epoch = 0  # first only
    # Get session files
    ses = vedb_store.Session.from_folder(argv[1], raise_error=False, overwrite_user_info=True)
    world_time_file, world_vid_file = ses.paths['world_camera']
    eye_left_time_file, eye_left_file = ses.paths['eye_left']
    eye_right_time_file, eye_right_file = ses.paths['eye_right']

    input_folder = argv[1]
    if os.path.exists(os.path.join(input_folder, 'marker_times.yaml')):
        with open(os.path.join(input_folder, 'marker_times.yaml'), 'r') as file:
            marker_times = yaml.safe_load(file)
    else:
        marker_times = vedb_store.utils.specify_marker_epochs(input_folder)
    validation_times = os.path.join(input_folder,
                                    'marker_times.yaml')
    eye_left_st, eye_left_end = vedb_store.utils.get_frame_indices(
                    *marker_times["calibration_times"][calibration_epoch], ses.get_video_time("eye_left"))
    eye_right_st, eye_right_end = vedb_store.utils.get_frame_indices(
                    *marker_times["calibration_times"][calibration_epoch], ses.get_video_time("eye_right"))

    return marker_times, eye_left_time_file, eye_left_file, world_time_file, world_vid_file, eye_right_time_file, eye_right_file,validation_times, eye_left_st, eye_left_end, eye_right_st, eye_right_end

def calibration_markers(world_vid_file, world_time_file, marker_times):
    print('Find calibration markers')
    calibration_markers = vedb_gaze.marker_detection.find_concentric_circles(world_vid_file, world_time_file, 
                                                            start_frame=marker_times['calibration_frames'][0][0], 
                                                            end_frame=marker_times['calibration_frames'][0][1], 
                                                            n_cores=n_cores,
                                                            progress_bar=tqdm)
def load_pylids(input_folder):
    with open(os.path.join(input_folder,'eye1.pkl'), 'rb') as f:
        pupil_left = pickle.load(f)
    with open(os.path.join(input_folder,'eye0.pkl'), 'rb') as f:
        pupil_right = pickle.load(f)
    print(pupil_left)
def head():
    odo = vedb_calibration.vedbCalibration()
    odo.set_odometry_local(argv[1])
    odo.start_end_plot()
    odo.t265_to_head_trans()
    print(odo.calib_odo)
def main():
    input_folder = argv[1]
    # pupil_left, pupil_right = pylids_run()
    marker_times, eye_left_time_file, eye_left_file, world_time_file, world_vid_file, eye_right_time_file, eye_right_file,validation_times, eye_left_st, eye_left_end, eye_right_st, eye_right_end = eye_utils()
    calibration_markers(world_vid_file, world_time_file, marker_times)
    load_pylids(input_folder)
if __name__ == "__main__":
    main()