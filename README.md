# Keyless Keyboard - A Virtual Chorded Keyboard

## Table of Contents
- [Keyless Keyboard](#keyless_keyboard)
  - [Related Work](#related_work)
  - [Methodology](#methodology)
    - [System Performance](#performance)
    - [Dataset Bias](#bias)
  - [Disability Justice Perspective](#justice)
  - [Learnings and Future Work](#learnings)
  - [App Accessibility](#accessibility)
  - [References](#references)
- [Usage](#usage)
    - [Environment Setup](#setup)
    - [Keyless Keyboard](#use_keyless)
    - [Data Collection Tool](#data_collection)
    - [Custom Model Training](#training)
    - [System Configuration](#config)

## Keyless Keyboard <a id="keyless_keyboard"></a>

Disabled users are often reliant on additional devices to perform the same tasks as able-bodied users, thus incurring additional costs. These costs may be significant. One such example is text entry. Blind and low-vision users may use speech-to-text rather than a keyboard, but this introduces privacy concerns when used around others. An alternative to speech-to-text is chorded keyboards. Chorded keyboards allow users to enter text using one-handed devices. Such devices allow users to enter text privately, but require users to spend additional money. These devices also support users with mobility impairments who find other text entry methods difficult.

The Keyless Keyboard is an experimental approach to providing an alternative to chorded keyboards. Rather than relying on an additional device that a user must purchase, the Keyless Keyboard uses web cameras to view a user’s hand and detect motion similar to that used on a chorded keyboard. Using a series of machine learning models, the Keyless Keyboard learns to detect finger motion in the form of tapping. Different finger tapping combinations will result in different keystrokes. A production-worth system could allow a user point a camera at their hand from any reasonable angle and accurately detect finger tapping combinations.

Web cameras are another device required to make the Keyless Keyboard work. However, web cameras are ubiquitous in society, readily available on smart phones and laptops. Chorded keyboard solutions typically cost in excess of $100 and can reach well over $200. Users without immediate access to a web camera may purchase lower end models for $15 on Amazon. Should a user require a new web camera to use the Keyless Keyboard, they must still incur a cost greater than abled users. While the Keyless Keyboard will not remove this barrier entirely, it will diminish it.

## Related Work <a id="related_work"></a>

Chorded Keyboards are a known technology that take a number of forms. What is common across these devices is the use of a single hand to enter keystrokes via keypress combinations. Chorded keyboards have been in use for over a century, predating computers. Original designs were intended for telegraphs and stenotypes [[1]](#1). The Hall Braille Typewriter, created in 1892, is one of the first known examples of a chorded keyboard intended for use by blind and low-vision users. This chorded keyboard acted as a typewriter for Braille. In the computer age, chorded keyboards have been adapter for use as keyboard input devices.

The Tap Keyboard [[2]](#2) is a device that resembles a series of rings worn on one hand. This device, using machine learning techniques, learns to identify finger tapping motions that are detected by the device. These motions are then registered as keystroke inputs. Like many chorded keyboards, the Tap Keyboard was not originally intended as a device geared towards disabled users. Rather, it was originally intended as a device to be used in virtual reality, allowing users to enter keystrokes without needing to see a physical keyboard. However, the device has proven useful as an alternative method of input for blind and low-vision users  [[3]](#3). Tap Keyboards are small and easily transported. As is common in technology intended to enable disabled users to access the same functionality as able-bodied users, Tap Keyboards are expensive. Entry-level devices retail for $135.

Research has been done on alternatives to physical input devices. Senorita [[4]](#4) is a virtual keyboard alternative, using a touchscreen interface rather than a physical device. Senorita displays chorded key entries on a touchscreen display. A user may press multiple key entries at a time, mimicking the input of a traditional chorded keyboard, to activate a keystroke. The research team behind Senorita performed usability testing with members of the blind and low-vision community. Usability testing found that on average, users were able to enter 14 words per minute using the Senorita interface, compared to 33 words per minute using a standard QWERTY keyboard.

## Methodology <a id="methodology"></a>

Keyless Keyboard relies on a series of machine learning models to identify the motion of a user’s hand. If the motion detected is one or more fingers tapping, the system determines the keystroke that should be activated.

Images are captured from a web camera and are passed through the hand detection pipeline. The system is made flexible by providing configuration parameters that control the number of video frames captured in a second and the number of milliseconds to capture data for. To avoid forcing users to conform to a set window for the capture of motion, the system also allows multiple windows to be configured and monitored concurrently. For example, a user may set the base configuration to 30 frames per second for 0.5 seconds, and may also configure the system to watch windows of 1 second, 1.5 seconds, and so on. Additionally, properly trained models used for finger detection will become robust to noisy datasets, such that motions that fall between preconfigured window lengths will still be properly detected.

![A diagram that depicts the flow of the Keyless Keyboard system. An image of a hand is passed into a hand recognition system that identifies the fingers active and determines the corresponding letter.](https://raw.githubusercontent.com/j-confusatron/tap_keyboard/main/img/final_arch.png "Webcam setup diagram")

The machine learning pipeline initially invokes a hand detection model implemented by the MediaPipe [[5]](#5) Python library. This model takes as input the raw image produced by the web camera. MediaPipe will detect the presence of a hand in the image, and will identify the points of the hand as x, y, z coordinates. This process is performed at every frame capture and an array of timesteps stores the MediaPipe output for each timestep, according to the length of the preconfigured window. For example, if the system is configured to capture 30 frames per second and the base window is configured for 500ms, the length of the timestep array is 15 steps of MediaPipe hand coordinates. This allows the system to capture the motion of the user’s hand, not just a single image of the hand. Additional configured windows for other lengths of time are set to use the same number of frame captures in their timestep arrays. Each window captures frames in an equally dispersed amount of time.

The output of the MediaPipe timestep capture is thus:

```[{(x,y,x) for 21 hand points} for FPS * (WINDOW_LENGTH / 1000) frame captures]```

During model training, the mean and standard deviation are learned independently for x, y, and z. In the production pipelines, these values are used to normalize the coordinate dimensions. The arrays are then flattened to a one-dimensional array.

Six logistic regression models are trained independently to identify on this flattened array to learn the following motions: a flat, neutral hand; a thumb tapping; an index finger tapping; a middle finger tapping; a ring finger tapping; a pinky finger tapping. The training dataset used contains examples of all possible finger tapping combinations, so that the models become robust to different tapping motions. 

If a model predicts a positive result, that is interpreted as that model’s finger tapping. Each model is assigned a point value. When a model predicts a tap, that model’s value is added to a sum. The sum then correlates to a keystroke. For simplicity, the experimental system uses a basic approach to assigning values, where each letter’s position in the alphabet is it’s required point value. A is 1, B is 2, and so on. Thumbs are assigned 1 point, index 2, middle 4, ring 8, and pinky 16. A neutral hand is assigned 0 points, which is interpreted as a space. If no model makes a positive prediction, then no keystroke is detected and the system will simply advance to the next frame. The points assigned to each finger and the points required for all letters are configurable.

### System Performance <a id="performance"></a>

To evaluate the performance of the machine learning system in identifying finger taps, each of the six logistic regression models are evaluated independently. The MediaPipe hand detection system is not evaluated, as it is beyond the control of this project.

A dataset for this project was created by over the course of roughly one hour, using a custom-built tool for data capture. The dataset was then split into model training (80%), hyperparameter validation (10%), and generalized test performance (10%). The overall dataset consisted of roughly 900 samples. Each sample took roughly 0.5 seconds to capture.

Validation system tuning involved using ROC and precision-recall curves to identify thresholds to use for each model independently. Each model is configured for a healthy balance of precision and recall.

| Model   | ROC AUC | Acc     | Precision | Recall  | F1      |
| ------- | ------- | ------- | --------- | ------- | ------- |
| Neutral | 0.99935 | 0.97849 | 0.95454   | 0.95454 | 0.95454 |
| Thumb   | 0.93492 | 0.89247 | 0.86842   | 0.86842 | 0.86842 |
| Index   | 0.99950 | 0.97849 | 0.97058   | 0.97058 | 0.97058 |
| Middle  | 0.96926 | 0.93548 | 0.93333   | 0.875   | 0.90322 |
| Ring    | 0.97972 | 0.91397 | 0.87179   | 0.91891 | 0.89473 |
| Pinky   | 0.99806 | 0.96774 | 1.0       | 0.91891 | 0.95774 |

An interesting and unintended consequence of this process was the small amount of time required to train an accurate system. This opens up he possibility of allowing users of the Keyless Keyboard to train their own, specialized models. Such a system would allow users to not use finger taps, but perhaps some other gesture better suited to their mobility.

### Dataset Bias <a id="bias"></a>

As with many machine learning systems, the potential for bias is a possibility with Keyless Keyboard. MediaPipe’s hand recognition models may include bias, such as a bias towards Caucasian hands. However, control over these models is outside of the scope for this project.

The dataset used to train the logistic regression models that recognize hand gestures is within the scope of this project. All training data is created by the author of this project. This data, when passed through MediaPipe, is distilled down to a series of x,y,x coordinates. This eliminates the potential for racial and gender bias. However, the potential for bias towards the author’s hand gestures is highly likely. It should be understood that the current models are biased towards able-bodied individuals with no missing fingers.

Due to time constraints of this project, the dataset is purposely limited to a single view angle. This limitation is unrealistic for release in a production-ready system. For experimental demonstration, this intended bias will suffice.

## Disability Justice Perspective <a id="justice"></a>

This project is motivated by three disability justice principles. The primary motivating principle is **anti-capitalist politics**. Rather than constructing societal systems which are accessible to all, disabled users are routinely expected to spend additional money on tools to meet their needs. This is true of chorded keyboard solutions, which typically retail for over $100. Conversely, the technology for Keyless Keyboard is free and only requires a web camera, which are readily available. Keyless Keyboard also address **cross-disability solidarity** by addressing multiple needs. Keyless Keyboard may support BLV users as a tactile solution, as well as users with limited mobility who find using full-sized keyboards difficult. Finally, Keyless Keyboard addresses the principle of **sustainability** by removing the need to purchase extra plastic devices that otherwise serve no useful purpose.

## Learnings and Future Work <a id="learnings"></a>

Keyless Keyboard’s experimental purpose is to demonstrate whether the application of a computer vision chorded keyboard is interesting and merits further research. In its current form, Keyless Keyboard is incapable of functioning as a production-ready system. In spite of having only a limited dataset from which to learn, Keyless Keyboard has demonstrated that it is capable of functioning at a high level, in terms of system accuracy and precision. Further, the discovery that only a small dataset is required to train Keyless Keyboard opens up the possibility that such a system may be highly customizable to a specific user’s needs. For example, a user with mobility impairment may prefer some other motion to finger tapping and could train the system to recognize their preferred input motion.

Further work on Keyless Keyboard should explore three areas. First, a production-capable model architecture is required. In it’s current iteration, Keyless Keyboard only supports a specific camera angle when viewing hands. A production system could expand upon this by introducing an additional pose-estimation model to determine the view angle of the camera on the hand, which could then delegate to pose-specific finger models. Such a system would need to be aware of the processing time of the pipeline. Every additional machine learning model will incur additional cost. Secondly, usability studies should be conducted to demonstrate the speed of the system. If Keyless Keyboard is incapable of producing sufficient words per minute, it may have no value. Finally, further study should be done on the system’s ability to learn from a limited dataset. The success of the initial small dataset is an exciting revelation that could prove useful in developing a highly customizable system.

## App Accessibility <a id="accessibility"></a>

Keyless Keyboard is not intended to be a stand-alone accessibility application. Rather, it is intended to be used as a tool to make applications accessible by providing accessible keyboard input. The current iteration is a stand-alone application that merely demonstrates the intelligent functionality of the tool. As an accessibility tool, Keyless Keyboard would likely take the form of a customizable virtual keyboard. For example, Android devices may install custom keyboards to be used in place of the standard, pre-installed Gboard. In such a scenario, Keyless Keyboard could be offered on the Google Play Store for free. Users would be free to download Keyless Keyboard and set it as their default keyboard or select it only when desired.

The Keyless Keyboard test application requires no keyboard and mouse interaction once running. The application responds only to finger tapping movements and responds with the character keypress associated with detected motion. Users may tap any finger combinations and the application will notify the user of the detected motion and its corresponding keypress.

## References <a id="references"></a>

<a id="1">[1]</a> 
https://en.wikipedia.org/wiki/Chorded_keyboard#History

<a id="2">[2]</a> 
https://www.tapwithus.com/product/tap-strap-2/

<a id="3">[3]</a> 
Jayaram Lamichhane (Sep 7, 2018). Wearable Technology and Blindness — A Look at the Tap Keyboard. https://medium.com/@jay_48065/wearable-technology-and-blindness-a-look-at-the-tap-keyboard-84edc6904cb1 

<a id="4">[4]</a> 
Gulnar Rakhmetulla; Ahmed Sabbir Arif (April 23, 2020). Senorita: A Chorded Keyboard for Sighted, Low Vision, and
Blind Mobile Users. https://dl.acm.org/doi/10.1145/3313831.3376576

<a id="5">[5]</a> 
https://google.github.io/mediapipe/

## Usage <a id="usage"></a>

### Environment Setup <a id="setup"></a>

1. Prerequisites: [Anaconda](https://www.anaconda.com/)
2. Clone this repo or download the latest [release](https://github.com/j-confusatron/tap_keyboard/releases).
3. Create a virtual environment. Open a CLI at the project root. Then enter:
  ```
  conda create --name keyless_keyboard python=3.9
  conda activate keyless_keyboard
  pip install -r requirements.txt
  ```

### Keyless Keyboard <a id="use_keyless"></a>

1. Open a CLI at the project root.
2. Activate the keyless_keyboard virtual environment: `conda activate keyless_keyboard`
3. Launch the tool: `python -m keyless_keyboard.keyless_keyboard`
4. The tool will take a moment to start.

### Data Collection Tool <a id="data_collection"></a>
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

### Custom Model Training <a id="training"></a>

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

### System Configuration <a id="config"></a>

All system configuration is stored in `/config/config.json`.

| Property        | Description                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| fps             | The number of frames of data per second.                                                                                        |
| capture_time    | The length of time (in milliseconds) to capture data for.                                                                       |
| num_windows     | The number of capture windows to capture data for evaluaion. Each window is capture_time ms longer than the previous.           |
| agreements      | The number of iterations that must produce a like result to register a keypress.                                                |
| num_hand_points | The data of MediaPipe hand capture points.                                                                                      |
| data_ratio      | An array of the sample ratio to be split. Element order is: train, validate, test. Must sum to 1.                               |
| mean            | The x,y,z data means used for normalization(set automatically by model_trainer.create_datasets).                                |
| stdev           | The x,y,z data standard deviation used for normalization(set automatically by model_trainer.create_datasets).                   |
| thresholds      | An array of the thresholds for (in order): neutral, thumb, index, middle, ring, pinky. Use ROC curves to inform these settings. |
| train           | True to train new models, else false to simply load existing models for eval.                                                   |
| evaluate        | True to run test set evaluations, else false.                                                                                   |
| scores          | Point values assigned to (in order): neutral, thumb, index, middle, ring, pinky. -1 is interpreted as 0.                        |
| keys            | Keypresses assigned to each value. Position in list correlates with score required, starting from 0.                            |