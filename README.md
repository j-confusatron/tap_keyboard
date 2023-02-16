# Keyless Keyboard

## Data Collection Tool
The goal of this project is to interpret finger taps as keyboard entries. Different finger tap combinations will result in different keystrokes. The tool will cycle through all possible combinations of finger-taps and ask you to record yourself performing that tap. The data captured by this tool will be used to train machine learning models that can identify fingers tapping.

### Tool Setup
1. Prerequisites: [Anaconda](https://www.anaconda.com/)
2. Clone this repo or download the latest [release](https://github.com/j-confusatron/tap_keyboard/releases).
3. Create a virtual environment. Open a CLI at the project root. Then enter:
  ```
  conda create --name keyless_keyboard python=3.9
  conda activate keyless_keyboard
  pip install -r requirements.txt
  ```

### Using the Tool

#### Preparing Your Space
This tool requires an available webcam. The current goal is to collect a very targeted dataset. This means the webcam will need to be in a specific position. Position your webcam above your desk, looking down on the area where your hand will be. If you have your webcam on your monitor, simply angling it down to look at the desk area in front of your monitor will work.

![diagram depicting a webcam on top of a monitor, angled down to view a hand resting on a desk](https://raw.githubusercontent.com/j-confusatron/tap_keyboard/main/cam_diagram.png "Webcam setup diagram")

#### Launch the Data Collection Tool
1. Open a CLI the project root.
2. Activate the keyless_keyboard virtual environment: `conda activate keyless_keyboard`
3. Launch the tool: `python -m data_collector.collector`
4. The tool will take a moment to start.

#### Using the Data Collection Tool

![Screenshot of the data collection tool. A hand is visible on a desk. Along the top, instructions are provided for a gesture to capture.](https://raw.githubusercontent.com/j-confusatron/tap_keyboard/main/data_collector_sample.png "Data Collection Tool")

| UI Element         | Position          | Purpose                                                                          |
| ------------------ | ----------------- | -------------------------------------------------------------------------------- |
| Grey Circle        | top left          | Grey indicates app is not recording. Red indicates app is recording              |
| Red X's, Green O's | top left          | Current sample. From left: thumb, index, middle, ring, pinky. O: tap; X: neutral |
| Pink Box           | main viewing area | Box and label of hand identified by app                                          |

At the top of the screen is a grey circle, followed by a series of X's and O's. The grey circle indicates that the tool is not currently recording. When the tool is recording, the circle turns red. The X's and O's indicate which fingers to tap. A red X indicates a finger should remain neutral, hovering an inch or so above the desk. A green O indicates that finger should tap on the desk. From left to right, the X's and O's indicate: thumb, index, middle, ring, pinky. Multiple indicated fingers should tap simultaneously. If only red X's are displayed, leave your hand in a neutral position about 1 inch above your desk for about 1 second while recording.

Only the **right hand** may be used to record motion.

**Steps**
1. Open Data Collection Tool
2. Observe the current sample (X's and O's)
3. Prepare to tap the fingers designated by O's
4. Press SPACE to start recording
5. Tap the indicated fingers
6. Press SPACE to stop recording
7. The tool will write the sample to disk and cycle to the next sample.

When you are ready to record, press the **SPACE** key to begin. Perform the tapping motion indicated. Then press **SPACE** again to stop recording. The data of your hand motion will be recorded and printed out on the CLI. The tool will then cycle to the next tap combination.

If you have made a mistake and wish to try again, press **BACKSPACE** to cycle back an entry. You may cycle back as many iterations as you wish. You may also press **BACKSPACE** while recording to cancel the recording.

When you are done collecting data, press **ESC** to exit the tool. You can then email me the data files generated during your testing. Message me on Git if you need my contact information.

| Key       | Purpose                                                                          |
| --------- | -------------------------------------------------------------------------------- |
| SPACE     | Start / Stop recording a sample                                                  |
| BACKSPACE | If app is recording, cancel current recording. Else, delete last recorded sample |
| ESC       | Exit application                                                                 |

### About the Data Collected
Every time the tool starts, an empty data file is created: `data/data_1234567890.json`. The data collected while recording finger tapping motions is stored in these files. The number at the end of the file name is the timestamp of when the tool was initiated.

Each sample recorded contains the following elements:
- y: the labels for the current sample (which fingers are actively tapping)
- x: the data points for the sample. This is an array where each item in the array represents a single timestep of the recorded motion. Each timestep consists of:
  - pts: the x,y coordinates of each point of the hand.
  - cent: the center coordinate of the hand.
  - box: the coordinates of the bounding box drawn around the hand.

I encourage everyone using this tool to look over the data. No machine or personal data is captured by this tool. No screenshots are captured by this tool.