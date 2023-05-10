import autofeat as aft
import numpy as np

if __name__ == '__main__':
    # Load data
    x = np.random(1000)
    
    # Create sliding window
    window = aft.SlidingWindow(window_size=100, step_size=100)

    # Create transform
    mean_tf = aft.MeanTransform()

    # Get featurizer
    featurizer = window.use(mean_tf)

    # Get features
    features = featurizer(x)

    # Print features
    print(window)
    print(mean_tf)
    print(features)


