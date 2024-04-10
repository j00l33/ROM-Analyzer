import tensorflow as tf
import numpy

def run_movenet(interpreter, frame, input_details, output_details):
    frame = tf.expand_dims(frame, axis=0)
    frame = tf.image.resize_with_pad(frame, 256, 256)
    frame = tf.cast(frame, dtype=tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def process_video(video_tensor, interpreter, input_details, output_details):
    total_keypoints_scores = []
    print("running movenet")
    for frame in video_tensor:
        keypoints_scores = run_movenet(interpreter, frame, input_details, output_details)
        total_keypoints_scores.append(keypoints_scores)

    print("movenet complete")
    return total_keypoints_scores