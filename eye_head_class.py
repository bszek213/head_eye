# Eye and Head analyss: Brian Szekely
import matplotlib.pyplot as plt
import numpy as np
import vedb_gaze
import vedb_store
from odopy import headCalibrate
import os
import yaml
from tqdm import tqdm
from sys import argv
import rigid_body_motion as rbm
# import pickle
n_cores = None 

class eyeHead():
    def __init__(self):
        print("instantiate eyeHead class object")
        self.input_folder = argv[1]
        self.fov_horiz = 125 #deg 
        self.fov_vertical = 110 #deg
        self.pupil_fs = 1/120
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
    def head_calibration(self):
        odo = headCalibrate.headCalibrate()
        odo.set_odometry_local(self.input_folder)
        odo.start_end_plot()
        odo.t265_to_head_trans()
        self.calib_odo = odo.get_calibrated_odo()
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
    def setup_ref_frame(self):
        #World Reference Frame
        rbm.register_frame("world", update=True)
        #Head Reference Frame
        rbm.register_frame("head",
                            translation=self.calib_odo.lin_pos,
                            rotation=self.calib_odo.ang_pos,
                            timestamps=self.calib_odo.time.values,
                            parent="world",
                            update=True,
                            )
        self.gaze_3d = np.array((self.gaze['left']['gaze_degrees'][:,0].reshape(-1), 
                       self.gaze['left']['gaze_degrees'][:,1].reshape(-1),
                       np.zeros((len(self.gaze['left']['gaze_degrees'][:,0])))
                       )).T
        print(self.gaze_3d)
        #mulitply the eye in world camera by the extrinsic rotation
        eye_orient = rbm.shortest_arc_rotation(self.gaze_3d,np.array((0, 0, 1)))
        print(eye_orient)
        # fig, ax = plt.subplots(1,2)
        # ax[0].plot(self.gaze['left']['timestamp'],self.gaze_3d[:,0],label='gaze x')
        # ax[1].plot(self.gaze['left']['timestamp'],eye_orient[:,1],label='gaze quaternion')
        plt.show()
        #Extrinsics reference frame
        rotation = [30,0,0] #degrees
        trans_y = 1.25 #inches
        trans_x = 0 #inches
        trans_z = 1.15 #inches
        #Eye reference frame
    def run_analysis(self):
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
        #RBM time
        self.setup_ref_frame()

def main():
    eyeHead().run_analysis()
if __name__ == "__main__":
    main()