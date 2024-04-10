# Rep and Range of Motion Tracker

This program is designed to be able to analyze a video of a given exercise, and inform the user of the amount of reps performed, and the relative range of motion (ROM) for each repetition. The entire program is written in python and utilizes the TensorFlow model MoveNet to detect the people and their respective joints from an image. The program requires two inputs, the video of the exercise being performed, and the selection of which exercise is being performed. If the exercise isn't available, a new instance of the exercise class can be created with specifications for the new movement. The program also displays a graph of joint angle and distance data for the video

## Structure

The program consists of 5 modules: 

main.py - Main module that handles project organization and user input.

video_input.py - Handles conversion of video into frame data that can be fed to the MoveNet model.

movenet.py - Handles inputting the frame data to the MoveNet model, and obtaining the output of the model.

analysis.py - Handles the analysis of the frame data and finding the repetitions and ROM. The bulk of the program is contained in this module.

exercises.py - Handles the definition and instantiations of the exercise class.

## Data analysis

The logic for the analysis module is as follows:

- Find the person in the video that is performing the exercise
- Single out that person in the dataset
- Find the measurements required for the exercise
- Apply a smoothing filter to the data
- Find the wave peaks in the key dataset
- Find the subset of the frame data where the exercise is being performed
- Eliminate the start and ending frame data if there is activity outside of the exercise being performed
- Calculate the repetitions and ROM based on the peaks and valleys of the cleaned dataset

## Notes

A new movement pattern can be defined in exercises.py by inputting the required angle and distance data.

On some machines, you may need to run 'set KMP_DUPLICATE_LIB_OK=TRUE' in order for the plotting function to run.

The specific model included is best ran on a PC, however, other models from MoveNet are offered which may better better suited for different environments.

Only accepts .mp4 files.