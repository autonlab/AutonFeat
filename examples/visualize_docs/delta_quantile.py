import sys
package_path = '/Users/dhruvsrikanth/Work/CMU/AutoFeat'
sys.path.append(package_path)

import numpy as np
import matplotlib.pyplot as plt
import autofeat as aft


def main():


    plt.tight_layout()

    # Save figure
    plt.savefig('../../docs/assets/delta_quantile_visualize.png')


if __name__ == '__main__':
    main()
