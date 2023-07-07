from autonfeat.core import SlidingWindow
from autonfeat.common import MeanTransform
import numpy as np


def test_fixed_window():
    overflow_methods = [
        'restrict',
        'pad',  # Note for padding we will consider the default value in testing i.e. signal is padded with zeros.
        'stop',
    ]
    pad_value = 0

    # Random window sizes and step sizes
    n_tests = 10
    window_sizes = [np.random.randint(1, 100) for _ in range(n_tests)]
    step_sizes = [np.random.randint(1, 100) for _ in range(n_tests)]

    # Random signal
    signal = np.random.rand(200)

    # Transform to test
    tf = MeanTransform()
    tf_hat = np.mean

    # Test
    for ws in window_sizes:
        for ss in step_sizes:
            for overflow in overflow_methods:
                window = SlidingWindow(window_size=ws, step_size=ss, overflow=overflow)
                featurizer = window.use(tf)

                y_hat = featurizer(signal)
                assert isinstance(y_hat, np.ndarray)

                # Compute y manually
                y = []
                for i in range(0, len(signal), ss):
                    if i + ws > len(signal):
                        if overflow == 'restrict':
                            y.append(tf_hat(signal[i:]))
                        elif overflow == 'pad':
                            overflow_size = i + ws - len(signal)
                            padded_signal = np.asarray(list(signal[i:]) + [pad_value] * overflow_size)
                            y.append(tf_hat(padded_signal))
                        elif overflow == 'stop':
                            break
                    else:
                        y.append(tf_hat(signal[i:i + ws]))

                # Check if the values are close to equal across the entire array
                assert np.allclose(y_hat, y)
