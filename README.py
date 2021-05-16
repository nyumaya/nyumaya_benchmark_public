
# Requirements

pip install tensorflow
pip install numpy
pip install pydub

# Running the benchmark

1. Set your data locations in config.py
2. Populate the noise folder with the demand dataset
3. Download the scenarios (sh download_scenarios.sh)
4. Run the benchmark python3 benchmark.py alexa 2.0.23
5. The results are written to the result_folder


# Scenarios

Scenarios are recording which do not contain the keyword. 
They are used for measuring false activations. For performance
reasons the audio is preprocessed and the extracted features are
stored in a tfrecord file. This way we can evaluate different scenarios
quickly and efficienlty.


## Adding own scenarios

An example for making a scenario is presented in the file makeSzenario.py
The scenario is composed of multiple recordings (called examples). Make sure
each example is not overly long for memory and performance reasons.


## Adding own keyword testdata

You should add your keyword files in the folder configured by the keyword_folder
parameter in a subfolder named after the keyword (eg. ./keywords/marvin).

