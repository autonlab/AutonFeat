import autofeat as aft
import numpy as np

if __name__ == '__main__':
    # Random data
    n_samples = 100
    x = np.random.rand(n_samples)
    
    # Create sliding window
    ws = 10
    ss = 10
    window = aft.SlidingWindow(window_size=ws, step_size=ss)

    # Create transform
    mean_tf = aft.MaxTransform()

    # Get featurizer
    featurizer = window.use(mean_tf)

    # Get features
    features = featurizer(x)

    # Print features
    print(window)
    print(mean_tf)
    print(features)

