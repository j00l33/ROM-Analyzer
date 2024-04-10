import tensorflow as tf
import cv2
import io
import imageio
import os

def load_video(video_path):
    max_size=(640, 480)
    frames = []
    
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, max_size, interpolation=cv2.INTER_AREA)
        frames.append(resized_frame)
        success, frame = video_capture.read()
    
    video_capture.release()

    gif_data = io.BytesIO()
    imageio.mimsave(gif_data, frames, format='GIF', fps=30)
    with open("output.gif", "wb") as f:
        f.write(gif_data.getvalue())

    print("Loading video\n")
    video = tf.io.read_file('./output.gif')
    print("Decoding video\n")
    video = tf.io.decode_gif(video)
    print("removal")


    os.remove('./output.gif')
 
    return video