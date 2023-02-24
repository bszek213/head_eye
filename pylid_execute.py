#Run pylids on folder path and save to .npz : Brian Szekely
import os
from sys import argv

#TODO: UPDATE FILE PATH TO SAVE NPZ TO INPUT_DIRECTORY
def pylids_run():
    import pylids
    import pickle
    #example: os.path.join("/home/bszekely/Desktop/eye_pipeline/test_data","eye1.mp4")
    #save_vid=True if you want the annotations
    print('Perfrom pylid analysis')
    pupil_left = pylids.analyze_video(eye_vid=os.path.join(argv[1],"eye1.mp4"),
                                      save_npz=True,
                                model_name='eyelids_pupils_v2')
    pupil_right = pylids.analyze_video(eye_vid=os.path.join(argv[1],"eye0.mp4"), 
                                       save_npz=True,
                                model_name='eyelids_pupils_v2'
                                )
    #Save data to binary file
    #HIGHEST_PROTOCOL :  higher values generally resulting in more compact data but potentially slower serialization.
    # with open(os.path.join(argv[1],'eye1.pkl'), 'wb') as f:
    #     pickle.dump(pupil_left, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(argv[1],'eye0.pkl'), 'wb') as f:
    #     pickle.dump(pupil_right, f, protocol=pickle.HIGHEST_PROTOCOL)
def main():
    if os.path.exists(os.path.join(argv[1],'eye1.npz')):
        print('file exists, load in pkle data')
    else:
        pylids_run()
if __name__ == "__main__":
    main()