import math
from exercises import exercise
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

WINDOWSIZE = 3
MINIMUM_ROM = .2

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def find_target(keypoints_list, movement):
    target_person_keypoints = None
    frames = 0
    left_side = True
    j = len(keypoints_list)
    for frame_keypoints in keypoints_list:
        num_people = frame_keypoints.shape[1]
        #print(frame_keypoints.shape)
        frames += 1
        #print(str(frames) + ' / ' + str(j))
        for person_idx in range(num_people):
            person_keypoints = frame_keypoints[0, person_idx]
            target_found = True
            #print(num_people)
            for joint_triplet, angle_range in movement.start_angles:
                joint1, joint2, joint3 = joint_triplet

                left_angle = calculate_angle(person_keypoints[joint1], person_keypoints[joint2], person_keypoints[joint3])
                #print('left: ' + str(left_angle))
                if angle_range[0] <= left_angle <= angle_range[1]:
                    left_side = True
                    continue
                else:
                    target_found = False
                    break
            
            if target_found:
                target_person_keypoints = person_keypoints
                break

            target_found = True
            for joint_triplet, angle_range in movement.start_angles:
                joint1, joint2, joint3 = joint_triplet
                right_joint1 = joint1 + 1
                right_joint2 = joint2 + 1
                right_joint3 = joint3 + 1 

                right_angle = 360 - calculate_angle(person_keypoints[right_joint1], person_keypoints[right_joint2], person_keypoints[right_joint3])
                #print('right: ' + str(right_angle))
                if angle_range[0] <= right_angle <= angle_range[1]:
                    #print('hit')
                    left_side = False
                    continue
                else:
                    target_found = False
                    break

            if target_found:
                target_person_keypoints = person_keypoints
                break

        if target_person_keypoints is not None:
            break

    return target_person_keypoints, frames, left_side

def find_closest_person(target_keypoints, frame_keypoints):
    num_people = frame_keypoints.shape[1]
    min_distance = float('inf')
    closest_person_idx = None

    main_joints = [KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['right_shoulder'],
                   KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['right_hip']]

    for person_idx in range(num_people):
        person_keypoints = frame_keypoints[0, person_idx]  
        total_distance = 0

        for joint_idx in main_joints:
            target_joint = target_keypoints[joint_idx][:2]  
            person_joint = person_keypoints[joint_idx][:2]  
            total_distance += math.dist(target_joint, person_joint)

        if total_distance < min_distance:
            min_distance = total_distance
            closest_person_idx = person_idx

    return frame_keypoints[0, closest_person_idx]  

def eliminate_other_people(keypoints_list, target_person_keypoints, target_frame_idx):
    filtered_keypoints = []

    filtered_keypoints.append(target_person_keypoints)

    for frame_idx in range(target_frame_idx + 1, len(keypoints_list)):
        frame_keypoints = keypoints_list[frame_idx]  
        closest_person = find_closest_person(target_person_keypoints, frame_keypoints)  
        filtered_keypoints.append(closest_person)
        target_person_keypoints = closest_person  

    return filtered_keypoints

def extract_relevant_data(single_target_keypoints, exercise, left_side):
    angles_data = []
    distances_data = []

    for angle_triplet in exercise.measure_angles:
        angle_list = []

        joint1, joint2, joint3 = angle_triplet

        if left_side == False:
            joint1 += 1
            joint2 += 1
            joint3 += 1
        
        for frame_keypoints in single_target_keypoints:
            angle = abs(calculate_angle(frame_keypoints[joint1], frame_keypoints[joint2], frame_keypoints[joint3]))
            if left_side:
                angle = 360 - angle
            angle_list.append(angle)

        angles_data.append(angle_list)

    for joint_pair in exercise.measure_distances:
        distance_list = []
        joint1, joint2 = joint_pair

        if left_side == False:
            joint1 += 1
            joint2 += 1

        for frame_keypoints in single_target_keypoints:
            distance = math.dist(frame_keypoints[joint1][:2], frame_keypoints[joint2][:2])
            distance_list.append(distance)

        distances_data.append(distance_list)

    return angles_data, distances_data

def moving_average_filter(data, window_size):
    data = np.array(data)

    window = np.ones(window_size) / window_size

    smoothed_data = np.convolve(data, window, mode='same')

    return smoothed_data

def detect_wave_peaks(data, prominence=None, distance=None):
    data = np.array(data)

    peaks, _ = find_peaks(data, prominence=prominence, distance=distance)

    return peaks.tolist()

def plot_data(angles_data, distances_data):
    num_frames = len(angles_data[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i, angle_list in enumerate(angles_data):
        angle_label = f"Angle {i + 1}"
        ax1.plot(range(num_frames), angle_list, label=angle_label)

    ax1.set_title("Joint Angles")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Angle (degrees)")
    ax1.legend()

    for i, distance_list in enumerate(distances_data):
        distance_label = f"Distance {i + 1}"
        ax2.plot(range(num_frames), distance_list, label=distance_label)

    ax2.set_title("Joint Distances")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Distance")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def identify_valid_range(peaks, difference_threshold):
    start = 0
    end = 1

    range_start, range_size = 0, 0
    while end < len(peaks):
        if(abs(peaks[end] - peaks[end - 1]) <= difference_threshold):
            end += 1
            if len(peaks) == end:
                range_size = end - start - 1
        else:
            if range_size < end - start - 1:
                range_size = end - start - 1
                range_start = start
            start = end 
            end += 1

    return range_start, range_size + range_start

def find_ends(frames, start_peak, end_peak):
    left_max = frames[start_peak]
    right_max = frames[end_peak]

    while start_peak >= 0:
        if frames[start_peak] > left_max:
            break
        else:
            start_peak -= 1

    while end_peak < len(frames):
        if frames[end_peak] > right_max:
            break
        else:
            end_peak += 1

    if start_peak < 0:
        start_peak = 0

    if end_peak >= len(frames):
        end_peak = len(frames) - 1

    return start_peak, end_peak

def find_peak_values(peaks_idx, dataset):
    peak_values = []
    for idx in peaks_idx:
        peak_values.append(dataset[idx])

    return peak_values

def find_rangeofmotion(dataset, peak_height):
    valley_idx_pairs = []
    local_start = 0
    valley = False if dataset[0] >= peak_height else True
    for x in range(len(dataset) - 1):
        if valley:
            if dataset[x] >= peak_height:
                valley = False
                valley_idx_pairs.append([local_start, x])
                continue
            if x == len(dataset) - 2:
                valley_idx_pairs.append([local_start, x])
        else:
            if dataset[x] < peak_height:
                valley = True
                local_start = x

    valley_mins = [min(dataset[pair[0]:pair[1]]) for pair in valley_idx_pairs]
    rom_score_heights = [peak_height - low for low in valley_mins]
    rom_scores_percentile = [height / max(rom_score_heights) for height in rom_score_heights]

    return rom_scores_percentile
    
def process_data(keypoints, movement):
    # Identify the target, the frame where identiied, and the targets orientation
    target_keypoints, target_frame_idx, left_side = find_target(keypoints, movement)
    
    # Eliminate the non-targets from the dataset
    single_target_keypoints = eliminate_other_people(keypoints, target_keypoints, target_frame_idx)
    print(str(len(single_target_keypoints)))
    
    # Find the required joint angles and distances of the targets keypoints for each frame
    angles, distances = extract_relevant_data(single_target_keypoints, movement, left_side)
    side = 'left' if left_side == True else 'right'
    print(side)

    # Apply smoothing filter to each angle and distance sublist
    angles_smooth = [moving_average_filter(angle_list,WINDOWSIZE) for angle_list in angles]
    distances_smooth = [moving_average_filter(distance_list,WINDOWSIZE) for distance_list in distances]

    # Find the indices and values of the peaks in the required distance or angle data
    if movement.primary_angle:
        peaks_idx = detect_wave_peaks(angles_smooth[movement.primary_angle_idx], 20, 20)
        peak_values = find_peak_values(peaks_idx, angles_smooth[movement.primary_angle_idx])
    else:
        peaks_idx = detect_wave_peaks(distances_smooth[movement.primary_distance_idx], 20, 20)
        peak_values = find_peak_values(peaks_idx, angles_smooth[movement.primary_angle_idx])

    # Find the indices of the valid start and end peaks    
    range_start, range_end = identify_valid_range(peak_values, 10)
    
    # Find the indices of the final frame range
    if movement.primary_angle:
        first_frame_idx, last_frame_idx = find_ends(angles_smooth[0], peaks_idx[range_start], peaks_idx[range_end])
    else:
        first_frame_idx, last_frame_idx = find_ends(distances_smooth[0], peaks_idx[range_start], peaks_idx[range_end])

    # Excise the ends of all the angle and distance data
    angles_final = [angle_list[first_frame_idx+1:last_frame_idx-1] for angle_list in angles_smooth]
    distances_final = [distance_list[first_frame_idx+1:last_frame_idx-1] for distance_list in distances_smooth]

    #print(peaks_idx)
    #print(str(first_frame_idx))
    #print(str(last_frame_idx))
    #print(peak_values)


    if movement.primary_angle:
        rom_scores_percentile = find_rangeofmotion(angles_final[movement.primary_angle_idx], min(peak_values))
    else:
        rom_scores_percentile = find_rangeofmotion(distances_final[movement.primary_distance_idx], min(peak_values))
        
    rom_scores_percentile = find_rangeofmotion(angles_final[0], min(peak_values))

    print(rom_scores_percentile)
    plot_data(angles_final, distances_final)

    return rom_scores_percentile
