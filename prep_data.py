import numpy as np
from scipy.ndimage.interpolation import zoom

def downsample_video(video, output_frames):
    from scipy.ndimage.interpolation import zoom
    output_vid = zoom(video, [(float(output_frames)/video.shape[0]),1,1,1])
    return output_vid

def read_video_to_frames(filename = 'train_001.avi', output_resolution = (150,150)):
    import imageio
    vid = imageio.get_reader(filename,  'ffmpeg')
    vid2 = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    no_of_frames = vid.get_length()
    input_resolution = np.array(vid2.get_data(0)).shape
    #frame_jumpsize = int(no_of_frames/n_frames)
    #print frame_jumpsize
    for i in range(0,no_of_frames,1):
        frames.append(zoom(np.array(vid.get_data(i)), [(output_resolution[0]/float(input_resolution[0])),(output_resolution[1]/float(input_resolution[1])),1]))
    frames = np.array(frames)
    print frames.shape
    return frames
        
# returns a numpy array pretty much ready for training as X_train
def get_video_ready_for_training(i=0, root = 'Left', output_resolution = (150,150)):
    import os
    import cv2
    face_detection_xml ="opencv2_data/haarcascades/haarcascade_frontalface_default.xml"
    faceDetectClassifier = cv2.CascadeClassifier(face_detection_xml)
    samples = os.listdir(root)
    Frames = read_video_to_frames(filename = os.path.join(root, samples[i]), output_resolution = output_resolution)
    Frames = downsample_video(Frames, 2400)
    x_train = np.zeros(shape = Frames.shape)
    i = 0
    for frame in Frames:
        facePoints = faceDetectClassifier.detectMultiScale(frame[:,:,0])
        if (len(facePoints)<1):
        	x_train[i] = frame
        	i += 1
        	continue
        x,y,w,h = facePoints[0]
        cropped = frame[y:y+h, x:x+w, :]
        face = zoom(cropped, [float(output_resolution[0])/cropped.shape[0],float(output_resolution[1])/cropped.shape[1], 1])
        #face_3d = face[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        x_train[i] = face
        i += 1
    x_train /= 255
    return x_train
