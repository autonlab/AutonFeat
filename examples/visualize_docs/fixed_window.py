import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    # Generate the sample signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * t) + np.cos(3 * t) + np.sin(5 * t) + np.cos(7 * t) + np.exp(-t / 5)

    # Sliding window parameters
    window_size = 50
    step_size = 25

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(signal))
    ax.set_ylim(-6, 6)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Signal Value', fontsize=12)
    ax.set_title('Sliding Window Feature Extraction', fontsize=14)

    # Customize the plot style
    plt.style.use('seaborn')

    # Initialize the line for the original signal
    line_original, = ax.plot([], [], color='steelblue', lw=2, label='Signal')

    # Initialize the scatter plot for the mean values
    scatter_mean, = ax.plot([], [], 'o', color='tab:orange', lw=2, alpha=0.8, label='Mean Features')

    # Initialize the rectangle patch for the sliding window
    rect = plt.Rectangle((0, -5.8), window_size, 11.6, edgecolor='black', facecolor='tab:orange', alpha=0.5, label='Window')
    ax.add_patch(rect)

    # Empty lists to store the mean values and corresponding x-positions
    mean_values = []
    x_positions = []

    # Function to update the animation
    def update(i):
        # Calculate the indices for the sliding window
        start = i * step_size
        end = start + window_size

        # Update the original signal line with new data
        line_original.set_data(np.arange(len(signal)), signal)

        # Extract the features within the sliding window
        features = signal[start:end]

        # Calculate the mean value of the features
        mean_value = np.mean(features)

        # Append the mean value and x-position
        mean_values.append(mean_value)
        x_positions.append(start + window_size // 2)

        # Update the scatter plot with all mean values
        scatter_mean.set_data(x_positions, mean_values)

        # Update the rectangle position
        rect.set_x(start)

        return line_original, scatter_mean, rect

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(len(signal) // step_size), interval=500, blit=True)

    # Add legend
    ax.legend(loc='upper right')

    # Save the animation as a GIF file
    ani.save('../../docs/assets/fixed_sliding_window_animation.gif', writer='pillow')


if __name__ == '__main__':
    main()
