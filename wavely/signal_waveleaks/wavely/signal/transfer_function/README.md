# Transfer function
This module is dedicated to the design and application of transfer functions between microphones. A transfer function currently allows one to transform a magnitude spectrum recorded with a source microphone into a magnitude spectrum that would have been recorded with a target microphone. The transfer is only implemented in the frequency domain for now.

## File tree
Organization of the package is the following:

- `wavely/signal/transfer_function/`: function collection used to apply the transfer function on real data:
	- impulse_response.py: functions used to measure and extract impulse responses.
	- transfer_function.py: function used to apply the transfer function on spectra.
	- utils.py: various utility functions.

## Determining the transfer function between two microphones

The transfer function from a microphone to another. To do this, people involved in these measurements are Pierre Méresse ("deconvolution") and Nicolas Côté ("amplitude_measurement"). The idea is to simultaneously measure a test signal with 2 microphones. Then, a postprocessing (that depends on the measurement method) is done in order to extract the transfer function.

Two different methods are used to compute the transfer function. The first method, called "deconvolution", uses the deconvolution between recorded signal and emitted signal, allowing to estimate the impulse response of the acquisition chain. The transfer function is then computed as the ratio of the frequency responses of the different microphones. The second method, called "amplitude_measurement", uses directly discrepancies between the amplitudes measured by each microphone.

After the measurements are performed, one can adapt the ipython notebooks in the `notebooks/transfer_function/'method-name'` directories in order to estimate the transfer function.

## Deconvolution (contact Pierre Méresse)

The idea is to convolve the received signal by the time reversed emission signal. (See, e.g. https://en.wikipedia.org/wiki/Deconvolution)

Important ! The emission signal must be saved as well as the received signal.

The used emission signals were chirps with the following properties:
- duration from 5s to 30s;
- spanning frequency from 2kHz to 50kHz.

During postprocessing, part of the impulse response is extracted on a time window that must be specified by the user. In practice, multiple windows boundaries are used, and the estimated frequency response corresponds to the average over the responses associated with the different windows. The transfer function is considered valid on a limited frequency domain, where variations are deemed 'low'.

## Amplitude measurement (contact Nicolas Côté)

TODO: Nicolas should fill this part.

## Measurement conditions

- Microphones are placed inside the anechoic "egg".
- The used emitter was a tweeter (reference ?).
- The use of multiple emitters (tweeter + loudspeaker) decreased result quality and was not used.

## What are the limitations of these methods ?

- Deconvolution: As the transfer function is reconstructed only on a portion of the frequency domain, the transfer function is valid in this domain but cannot be used to reconstruct a time signal. This limitation could be overcome if the transfer function could be measured on a frequency domain larger than the signal spectrum.

- Amplitude measurement: only valid in the frequency domain and no possibility to apply it in time domain.

## How to integrate a new microphone couple in the package ?

After having determined the new transfer function, the notebook returns a set of coefficients allowing to compute a polynomial approximating the experimental transfer function. This set must be store in the `settings.yaml` file under the key `transfer_parameters`, then`'measurement_method'`, `coef_micA_to_micB`.

The subkeys are:
- "weight": the polynomial weights, as computed by `np.polyfit`,
- "domain": transfer function frequency validity domain.

Then, the `tf_coefficients_file` dictionary (in the `utils` submodule) must be updated with the new microphone couple.


# Applying a transfer function to arbitrary signals

The module offers several function to apply the transfer function.

## On a single value

Use the function `transfer_function.transform_LEQ_value(...)` on a single value.

## On a pd.DataFrame

Use the function `transfer_function.transform_dataframe_LEQ(...)` on the feature dataframe.

## On a full dataset

A full dataset is defined as collection of .wav files along with associated metadata.

First, L_eq levels must be computed band-wise with signal. This can be done through the `datasets-analysis` package. The function `batch_transform_dataframe_LEQ()`, in `datasets-analysis.preprocessing.transform` submodule can be used to transform the whole dataset. A dataframe associated with an additionnal column corresponding to the transformed bands is saved in a new .h5 file.

# Module limitations

The transfer can only be applied on L_eq bands, as the transfer function is applied on a feature dataframe compute with the `signal` module. An important improvement of the module will be to be able to get the full impulse response of a system, that could be applied beforehand on time signals, in order to compute any `signal` available feature on transformed signals.
