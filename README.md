# Keyless Keyboard
The Keyless Keyboard is an alternative to [chorded keyboards](https://en.wikipedia.org/wiki/Chorded_keyboard). Chorded keyboards can be expensive to purchase and require a specialized device to use. The Keyless Keyboard requires only a webcam to use. Webcams are ubiquitous and cheap. Many people already have one, either on their laptop or cell phone. By capturing images from a webcam, the Keyless Keyboard will recognize finger tapping gestures and translate them to key presses.

Keyless Keyboard will use a method similar to chorded keyboards, where different finger tap combinations are translated to specific key presses. Each finger is assigned a point value.
- Thumb: 1
- Index: 2
- Middle: 4
- Ring: 8
- Pinky: 16

By summing the total of the fingers tapped, the letter to be entered is determined.

The goal for this research project is to produce a key entry system that is accessible to blind and low vision users. This system may also benefit users with impaired mobility.

As this is a research project with a limited timeline, the scope of this project is purpoosely reduced. The dataset used to train the system will be small and targeted to a specific viewing angle. A more robust, production-ready system would be capable of recognizing input from any viewing angle.

## Usage

### Environment Setup

1. Prerequisites: [Anaconda](https://www.anaconda.com/)
2. Clone this repo or download the latest [release](https://github.com/j-confusatron/tap_keyboard/releases).
3. Create a virtual environment. Open a CLI at the project root. Then enter:
  ```
  conda create --name keyless_keyboard python=3.9
  conda activate keyless_keyboard
  pip install -r requirements.txt
  ```

### Keyless Keyboard

1. Open a CLI at the project root.
2. Activate the keyless_keyboard virtual environment: `conda activate keyless_keyboard`
3. Launch the tool: `python -m keyless_keyboard.keyless_keyboard`
4. The tool will take a moment to start.

### Data Collection Tool
The goal of this project is to interpret finger taps as keyboard entries. Different finger tap combinations will result in different keystrokes. The tool will cycle through all possible combinations of finger-taps and ask you to record yourself performing that tap. The data captured by this tool will be used to train machine learning models that can identify fingers tapping.

#### Preparing Your Space
This tool requires an available webcam. The current goal is to collect a very targeted dataset. This means the webcam will need to be in a specific position. Position your webcam above your desk, looking down on the area where your hand will be. If you have your webcam on your monitor, simply angling it down to look at the desk area in front of your monitor will work.

![diagram depicting a webcam on top of a monitor, angled down to view a hand resting on a desk](https://raw.githubusercontent.com/j-confusatron/tap_keyboard/main/img/cam_diagram.png "Webcam setup diagram")

#### Launch the Data Collection Tool
1. Open a CLI at the project root.
2. Activate the keyless_keyboard virtual environment: `conda activate keyless_keyboard`
3. Launch the tool: `python -m data_collector.collector`
4. The tool will take a moment to start.

#### Using the Data Collection Tool

![Screenshot of the data collection tool. A hand is visible on a desk. Along the top, instructions are provided for a gesture to capture.](https://raw.githubusercontent.com/j-confusatron/tap_keyboard/main/img/data_collector_sample.png "Data Collection Tool")

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
6. Recording will end after 1 second
7. The tool will write the sample to disk and cycle to the next sample.

When you are ready to record, press the **SPACE** key to begin. Perform the tapping motion indicated. The data of your hand motion will be recorded and printed out on the CLI. The tool will then cycle to the next tap combination.

If you have made a mistake and wish to try again, press **BACKSPACE** to cycle back an entry. You may cycle back as many iterations as you wish. You may also press **BACKSPACE** while recording to cancel the recording.

When you are done collecting data, press **ESC** to exit the tool. You can then email me the data files generated during your testing. Message me on Git if you need my contact information.

| Key       | Purpose                                                                          |
| --------- | -------------------------------------------------------------------------------- |
| SPACE     | Start / cancel recording a sample                                                  |
| BACKSPACE | If app is recording, cancel current recording. Else, delete last recorded sample |
| ESC       | Exit application                                                                 |

#### About the Data Collected
Every time the tool starts, an empty data file is created: `data/data_1234567890.json`. The data collected while recording finger tapping motions is stored in these files. The number at the end of the file name is the timestamp of when the tool was initiated.

Each sample recorded contains the following elements:
- y: the labels for the current sample (which fingers are actively tapping)
- x: the data points for the sample. This is an array where each item in the array represents a single timestep of the recorded motion. Each timestep consists of:
  - pts: the x,y coordinates of each point of the hand.
  - cent: the center coordinate of the hand.
  - box: the coordinates of the bounding box drawn around the hand.

This tool does not capture:
- machine data
- personal data
- screenshots

### Custom Model Training

This repo provides pre-trained binary classifiers for each finger and the neutral hand position. Models may be found under `/models`.

In order to train custom models, custom data must be generated by the Data Collection Tool. The Data Collection Tool will save this data under `/data`. Note that model training supports 1-* data_timestamp.json files in this directory. Data will be gathered from all available files for training.

Model training is a two step process. First, the raw data must be split into train, validate, and test datasets. Next, the models will be trained on the split data.
1. Open a CLI at the project root.
2. Activate the keyless_keyboard virtual environment: `conda activate keyless_keyboard`
3. Split the datasets: `python -m model_trainer.create_datasets`
4. Train the models: `python -m model_trainer.trainer`

At model training completion, the following data is saved:
- Models saves to `/models`
- ROC & Precision-Recall curves saved to `/metrics/curves.txt`
- Test metrics saved to `/metrics/metrics.txt`

#### Configuring Model Training

All system configuration is stored in `/config/config.json`. The relevant properties for model training:

| Property     | Description                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| fps          | The number of frames of data per second.                                                                                        |
| capture_time | The length of time (in milliseconds) to capture data for.                                                                       |
| data_ratio   | An array of the sample ratio to be split. Element order is: train, validate, test. Must sum to 1.                               |
| thresholds   | An array of the thresholds for (in order): neutral, thumb, index, middle, ring, pinky. Use ROC curves to inform these settings. |
| train        | True to train new models, else false to simply load existing models for eval.                                                   |
| evaluate     | True to run test set evaluations, else false.                                                                                   |