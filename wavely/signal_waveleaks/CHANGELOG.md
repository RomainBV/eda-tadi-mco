## v0.13.0 (2025-08-22)

### Feat

- add make linear filter function (iir/fir)

## 0.12.2 (2024-07-05)

### Fix

- scipy compatibility (bis)

## 0.12.1 (2024-07-05)

### Fix

- scipy compatibility for vib features

## v0.12.0 (2024-01-30)

### Feat

- mic simulations
- add workflow to build and upload package

## v0.11.0 (2023-03-28)

### Feat

- Threshold event detetion
- Histogram computation (FadingHistogram, SlidingHistogram, MergedHistogram)
- Remove surrounding time in event detection

### Fix

- Change Hysteresis in temporal event detection

### Refactor

- Temporal event detection

## 0.10.2 (2023-02-01)

### Fix

- make the package PEP561 compliant

## 0.10.1 (2023-01-30)

### Fix

- python_require format in setup

## v0.10.0 (2023-01-24)

### Feat

- add thresholdEventdetector and hysteresis function
- add compensate mems filter

## v0.9.0 (2022-03-24)

### Feat

- **preprocessing**: add C-Weighting filter
- check compatibility with Python3.9
- **helpers**: add a SignalSplitter class

### Fix

- **band_indexes**: missing check for low sample rate

### Refactor

- optimize import time

## v0.8.3 (2021-05-11)

## v0.8.2 (2021-04-26)

### Feat

- **preprocessing**: implement an overlap and add filtering function
- **quality**: add signal quality checks
- **vibrationfeatures**: change to IS unit base and add converters

### Refactor

- **vibrationfeatures**: change default value of filter output in gl_filter

## v0.8.1 (2021-02-16)

### Fix

- **preprocessing**: allow to choose filtering operation

## v0.8.0 (2020-11-24)

### Feat

- **vibrationfeature**: add `filter_output` argument in `gl_filter`
- add two new microphone transfer functions
-
    * data: add microphone characterization data from Bouygues
    * settings: add the frequency responses weights of the two vesper groups
    * transfer_function: add the microphone names to utils

Closes: #105

### Refactor

- **notebooks**: move notebooks to audio-signal-processing-experiments

## v0.7.1 (2020-10-29)

### Fix

- **setup**: resolve pip 2020 resolver
- tests, pylama, isort, and notebooks

## v0.7.0 (2020-09-15)

## v0.6.3 (2020-09-07)

## v0.6.2 (2020-07-29)

## v0.6.1 (2020-07-06)

## v0.6.0 (2020-06-10)

## v0.5.2 (2020-06-01)

## v0.5.2.dev0 (2020-05-29)

## v0.5.1 (2020-05-14)

## v0.5.0 (2020-04-09)

## v0.4.1 (2020-03-17)

## v0.4.0 (2020-03-16)

## v0.3.0 (2020-01-16)

## v0.2.0 (2019-12-12)

## v0.1.0 (2019-11-26)
