from itertools import cycle
from typing import Optional, Tuple, List

import numpy as np

from .transforms import CompressTransform, StretchTransform, AnomalyTransform

InjectionResult = Tuple[np.ndarray, np.ndarray]
"""Type alias for the result of the anomaly injection function."""
        
def inject_anomalies(
    data: np.ndarray,
    anomaly_types: List[str] = ["outlier", "compress", "stretch", "noise", "smoothing", "hmirror", "vmirror", "scale", "pattern"],
    max_anomaly_ratio: float = 0.10,
    min_anomaly_length: int = 10,
    max_anomaly_length: int = 100,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose=False,
) -> InjectionResult:
    """Inject anomalies into a time series.

    Parameters
    ----------
    data : np.ndarray
        Time series.
    anomaly_types : List[str]
        Types of anomalies to inject. One of ``"outlier"``, ``"compress"``, ``"stretch"``, ``"noise"``,
        ``"smoothing"``, ``"hmirror"``, ``"vmirror"``, ``"scale"``, or ``"pattern"``.
    n_anomalies : int
        Number of anomalies to inject.
    anomaly_length : int
        Length of the anomaly to inject.
    random_state : int, optional
        Seed used by the random number generator.
    rng : np.random.RandomState, optional
        Random number generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of the time series data, anomaly labels
    """
    labels = np.zeros_like(data)
    
    if rng is None:
        rng = np.random.default_rng(random_state)
    rng.shuffle(anomaly_types)
    margin = max_anomaly_length // 4

    i = 0
    for anomaly_type in cycle(anomaly_types):
        if np.sum(labels) / len(labels) > max_anomaly_ratio:
            if verbose: print(f"Stopping anomaly injection\tanomaly_ratio: {np.sum(labels)/len(labels):.0%}.")
            break
        # strength
        strength = AnomalyTransform.sample_strength(anomaly_type, rng)
        # adjust length for stretch/compress anomalies to ensure that final anomaly length is the desired one
        anomaly_length = np.random.randint(low=min_anomaly_length, high=max_anomaly_length+1)
        if anomaly_type == "stretch":
            length = int(np.ceil(anomaly_length / StretchTransform.factor(strength)))
        elif anomaly_type == "compress":
            length = int(np.ceil(anomaly_length / CompressTransform.factor(strength)))
        else:
            length = anomaly_length

        # get anomaly position
        section_size = len(data) // 3
        position = len(data)
        max_tries = 100

        while (position + length > len(data) or np.any(labels[position-margin:position + length + margin])
        ) and max_tries > 0:
            position_idx = rng.choice([0, 1, 2], p=[0.3, 0.3, 0.4])
            position_within_section = rng.integers(0, max(1, section_size - length))
            position = position_idx * section_size + position_within_section
            max_tries -= 1
        if max_tries == 0:
            if verbose: print(f"Could not find a position for the anomaly {i+1} of type {anomaly_type}, skipping!")
            i += 1
            break

        # get anomaly subsequence
        anomaly_subsequence = data[position:position + length]
        anomaly_subsequence, label_subsequence = _anomaly_transformer(anomaly_type, strength, rng)(anomaly_subsequence)
        anom_length = len(anomaly_subsequence)

        if verbose:
            print(f"Injecting anomaly {i+1} of type {anomaly_type} at position {position} with strength "
                f"{strength:.2f} and length {anomaly_length}.")
        i += 1
        data = np.r_[data[:position], anomaly_subsequence, data[position + length:]]
        labels = np.r_[labels[:position], label_subsequence, labels[position + length:]]
        
    return data, labels


def _anomaly_transformer(anomaly_type: str, strength: float, rng: np.random.Generator) -> AnomalyTransform:
    return AnomalyTransform.get_factory(anomaly_type)(strength=strength, rng=rng)