import sys
import tensorflow as tf
from video_input import load_video
from movenet import process_video
from analysis import process_data 
from exercises import exercise, movements

#set KMP_DUPLICATE_LIB_OK=TRUE

def main(video_path, movement):
    video_tensor = load_video(video_path)

    interpreter = tf.lite.Interpreter(model_path='./4.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    keypoints_scores = process_video(video_tensor, interpreter, input_details, output_details)
    rangeofmotion_scores = process_data(keypoints_scores, movement)


if __name__ == "__main__":
    video_path = input("Enter the file path of the video: ")
    movement_code = input("Enter the movement code: ")
    for movement in movements:
        if movement.code == movement_code:
            main(video_path, movement)
