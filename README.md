
# Requirements

pip install tensorflow
pip install numpy
pip install pydub

# Running the benchmark

1. Set your data locations in config.py
2. Point nyumaya_basepath to the nyumaya_audio_recognition repository
 or clone it to the pointed path (./nyumaya_audio_recognition/)
3. Populate the noise folder with the demand dataset
4. Download the scenarios (sh download_scenarios.sh)
5. Ensure your model is in the nyumaya_audio_recognition/models/Hotword folder
6. Run the benchmark python3 benchmark.py alexa 2.0.23
7. The results are written to the result_folder


# Scenarios

Scenarios are recording which do not contain the keyword. 
They are used for measuring false activations. For performance
reasons the audio is preprocessed and the extracted features are
stored in a tfrecord file. This way we can evaluate different scenarios
quickly and efficienlty.

Currently two scenarios are availabled

1. libri_test_v1.0
Test part of the LibriTTS dataset(https://research.google/tools/datasets/libri-tts/)
English read speech

2. ambient_test_v1.0
Ambient Noises partly taken from public domain sounds of freesound. It consists
mostly of longer recordings of fireworks, beach sounds, rain, wind, shower etc.
No intelligle speech is present.

## Adding own scenarios

An example for making a scenario is presented in the file makeSzenario.py
The scenario is composed of multiple recordings (called examples). Make sure
each example is not overly long for memory and performance reasons.


## Adding own keyword testdata

You should add your keyword files in the folder configured by the keyword_folder
parameter in a subfolder named after the keyword (eg. ./keywords/marvin).

