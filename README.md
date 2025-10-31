# Custom-Sign-Language-Recognition
This project was created to honor the unique sign language developed for a deaf and mute family member. When they were a child, our family invented this custom sign language to help them communicate with others. Because it is unique, only members of my family have learned and used it over the years.

Currently, the model can recognize three hand gestures — Thank You, Hug, and Sleepy — which are among the most commonly used signs in our family.
The project was implemented using **OpenCV** and **MediaPipe** for hand detection and gesture recognition.

## Installation
This project was created and tested on a virtual environment using Anaconda. Open Anaconda Prompt and write the following lines.

### Create a new anaconda environment
```bash
conda create --name Custom_Sign_Language_env python=3.11
```

### Activate the environment
```bash
conda activate Custom_Sign_Language_env
```
### Packages
```bash
pip install --upgrade pip setuptools wheel
pip install opencv-python
pip install scikit-learn
pip install mediapipe
pip install pandas
```

## Deactivate an active environment
```bash
conda deactivate
```

## Notes
The main issue in this project was the data collection as the model would get confused between 'thank you' and 'sleepy' due to using one hand. Contrary to 'hug' that uses both. For the moment, the right hand is prioritized when saying 'thank you' or indicating you are 'sleepy'. While collecting the data, it is important to move the hands around and in different orientations to help the model differientiate the sign.

### Feature Improvements
Future improvements may include:
- Expanding the dataset to recognize more gestures.
- Adding functionality to avoid displaying a label when no sign is detected.
