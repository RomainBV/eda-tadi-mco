import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, sosfilt

from wavely.signal.features.features import feature, noexport
from wavely.signal.features.temporalfeatures import (
    crest,
    crest_factor,
    peak_to_peak,
    power,
    rms,
)
from wavely.signal.models import FilterOutput
from wavely.signal.units.converters import g_to_ms2, lin2db

############################
# VIBRATION BASED FEATURES #
############################

REF_ACCELERATION = 1e-6  # m/s²
REF_VELOCITY = 1e-9  # m/s


@noexport
@feature
def gl_filter(
    blocks: np.ndarray,
    rate: float,
    Wn: np.ndarray,
    order: int,
    filter_output: FilterOutput = FilterOutput.sos,
) -> np.ndarray:
    """Provide a filtered version of the input signal.

    Note:
        Beware of using`FilterOutput.ba` while blocks have small durations (e.g. 40 ms).
        `FilterOutput.sos` should be preferred in that case to avoid numerical problems.

    Args:
        blocks: An array of data (acceleration or velocity array) to be
            filtered.
        rate: The sampling rate.
        Wn: A scalar or length-2 sequence giving the critical frequencies.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        The filtered output with the same shape as array.

    """
    nyq = 0.5 * rate  # Nyquist frequency
    normal_cutoff = Wn / nyq  # Normalized frequency
    if filter_output == FilterOutput.ba:
        b, a = butter(order, normal_cutoff, btype="bandpass", analog=False)
        return filtfilt(b, a, blocks)
    elif filter_output == FilterOutput.sos:
        sos = butter(order, normal_cutoff, btype="bandpass", analog=False, output="sos")
        return sosfilt(sos, blocks)
    else:
        raise ValueError(
            '"{} is not a valid filter output. Please use a FilterOutput type"'.format(
                filter_output
            )
        )
    # TODO: create a generic filtering function in preprocessing so as to take into
    # account the filter type


@feature("vibration", dims=["block", "time"])
def acceleration(blocks: np.ndarray) -> np.ndarray:
    """Return the acceleration in m/s² from the acceleration input in g.

    Args:
        blocks: The acceleration array in g.

    Returns:
        The acceleration array in m/s².

    """
    return g_to_ms2(blocks)


@feature("vibration", dims=["block", "time"])
def velocity(acceleration: np.ndarray, rate: float) -> np.ndarray:
    """Compute velocity from an acceleration input.

    Args:
        acceleration: The acceleration array in m/s².
        rate: The sampling rate.

    Returns:
        The velocity array in m/s.

    """
    return cumtrapz(y=acceleration.flatten("C"), dx=1 / rate, initial=0).reshape(
        acceleration.shape
    )


@feature("vibration", dims=["block", "time"])
def displacement(velocity: np.ndarray, rate: float) -> np.ndarray:
    """Compute displacement from a velocity input.

    Args:
        velocity: The velocity array in m/s.
        rate: The sampling rate.

    Returns:
        The displacement array in m.

    """
    return cumtrapz(y=velocity.flatten("C"), dx=1 / rate, initial=0).reshape(
        velocity.shape
    )


@feature("vibration", dims=["block"])
def gl_acceleration(
    blocks: np.ndarray,
    rate: float,
    Wn_gla: np.ndarray,
    order: int,
    filter_output: FilterOutput,
) -> np.ndarray:
    """Compute the global level acceleration in g.

    Args:
        blocks: The acceleration array in g.
        rate: The sampling rate.
        Wn_gla: A scalar or length-2 sequence giving the critical
            frequencies for global acceleration level.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        Global level acceleration in g.

    """
    return rms(
        power(
            blocks=gl_filter(
                blocks=blocks,
                rate=rate,
                Wn=Wn_gla,
                order=order,
                filter_output=filter_output,
            )
        )
    )


@feature("vibration", dims=["block"])
def gl_velocity(
    velocity: np.ndarray,
    rate: float,
    Wn_glv: np.ndarray,
    order: int,
    filter_output: FilterOutput,
) -> np.ndarray:
    """Compute the global level velocity in m/s.

    Args:
        velocity: The velocity array in m/s.
        rate: The sampling rate.
        Wn_glv: A scalar or length-2 sequence giving the critical
            frequencies for global velocity level.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        Global level velocity in m/s.

    """
    return rms(
        power(
            blocks=gl_filter(
                blocks=velocity,
                rate=rate,
                Wn=Wn_glv,
                order=order,
                filter_output=filter_output,
            )
        )
    )


@feature("vibration", dims=["block"])
def peak_to_peak_displacement(
    displacement: np.ndarray,
    rate: float,
    Wn_glv: np.ndarray,
    order: int,
    filter_output: FilterOutput,
) -> np.ndarray:
    """Compute the peak-to-peak displacement (in m) over the frequency band Wn_glv.

    This indicator is recommended by the API (American Petroleum Institute). By
    default  Wn_glv=([10 - 1kHz]).

    Args:
        displacement: The displacement array in m.
        rate: The sampling rate.
        Wn_glv: A scalar or length-2 sequence giving the critical
            frequencies for global displacement level.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        The peak-to-peak displacement over the frequency band Wn_glv.

    References:
        [1] MOBLEY, R. Keith. Vibration fundamentals. Elsevier, 1999.
        [2] SCHEFFER, Cornelius et GIRDHAR, Paresh. Practical machinery vibration
            analysis and predictive maintenance. Elsevier, 2004.

    """
    return peak_to_peak(
        gl_filter(
            blocks=displacement,
            rate=rate,
            Wn=Wn_glv,
            order=order,
            filter_output=filter_output,
        )
    )


@feature("vibration", dims=["block"])
def crest_factor_acceleration(
    blocks: np.ndarray,
    gl_acceleration: float,
    rate: float,
    Wn_gla: np.ndarray,
    order: int,
    filter_output: FilterOutput,
) -> np.ndarray:
    """Compute the crest factor of acceleration (without unit).

    The crest factor is defined as the ratio between the crest value and the
    RMS over the frequency band Wn_gla ([1kHz - 10kHz]).

    Args:
        blocks: The acceleration array in g.
        gl_acceleration: The global level acceleration in g.
        rate: The sampling rate.
        Wn_gla: A scalar or length-2 sequence giving the critical
            frequencies for global acceleration level.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        Crest factor of acceleration.

    """
    return crest_factor(
        crest=crest(
            gl_filter(
                blocks=blocks,
                rate=rate,
                Wn=Wn_gla,
                order=order,
                filter_output=filter_output,
            )
        ),
        rms=gl_acceleration,
    )


@feature("vibration", dims=["block"])
def k_factor_acceleration(
    blocks: np.ndarray,
    gl_acceleration: float,
    rate: float,
    Wn_gla: np.ndarray,
    order: int,
    filter_output: FilterOutput,
) -> np.ndarray:
    """Compute the K factor of acceleration in g².

    The K factor is defined as the product of the crest value and the
    RMS over the frequency band Wn_gla ([1kHz - 10kHz]).

    Args:
        blocks: The acceleration array in g.
        gl_acceleration: The global level acceleration in g.
        rate: The sampling rate.
        Wn_gla: A scalar or length-2 sequence giving the critical
            frequencies for global acceleration level.
        order: Filter order.
        filter_output: A FilterOutput type.

    Returns:
        K factor of acceleration.

    """
    return (
        crest(
            gl_filter(
                blocks=blocks,
                rate=rate,
                Wn=Wn_gla,
                order=order,
                filter_output=filter_output,
            )
        )
        * gl_acceleration
    )


@feature("vibration", dims=["block"])
def acceleration_level(acceleration: np.ndarray) -> np.ndarray:
    """Compute the acceleration level relative to an acceleration reference.

    Args:
        acceleration: the acceleration array in m/s².

    Returns:
        The acceleration level in dB.

    """
    return 2.0 * lin2db(rms(power(acceleration)) / REF_ACCELERATION)


@feature("vibration", dims=["block"])
def velocity_level(velocity: np.ndarray) -> np.ndarray:
    """Compute the velocity level relative to a velocity reference.

    Args:
        velocity: the velocity array in m/s.

    Returns:
        The velocity level in dB.

    """
    return 2.0 * lin2db(rms(power(velocity)) / REF_VELOCITY)
