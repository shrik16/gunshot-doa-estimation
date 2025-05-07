import numpy as np
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Constants
SPEED_OF_SOUND = 343  # Speed of sound in m/s
SAMPLE_RATE = 22050   # Sampling rate in Hz
SIGNAL_DURATION = 0.02  # Duration of simulated signal in seconds
NOISE_LEVEL = 0.1     # Noise amplitude relative to signal
REVERB_REFLECTIONS = 3  # Number of reflections to simulate
REVERB_DECAY = 0.5    # Amplitude decay factor per reflection
REVERB_DELAY = 0.002  # Delay between reflections (seconds)

# Microphone array positions (tetrahedral configuration in meters)
MIC_POSITIONS = np.array([
    [0.0, 0.0, 0.0],        # Mic 1 (origin)
    [0.1, 0.0, 0.0],        # Mic 2 (x-axis)
    [0.05, 0.0866, 0.0],    # Mic 3 (xy-plane)
    [0.05, 0.0289, 0.0816]  # Mic 4 (above xy-plane)
])

# Simulate a gunshot signal (impulsive, wideband)
def generate_gunshot_signal():
    t = np.linspace(0, SIGNAL_DURATION, int(SAMPLE_RATE * SIGNAL_DURATION))
    signal = np.zeros_like(t)
    signal[:int(SAMPLE_RATE * 0.001)] = np.exp(-1000 * t[:int(SAMPLE_RATE * 0.001)])
    return signal, t

# Add noise to the signal
def add_noise(signal, noise_level=NOISE_LEVEL):
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

# Add reverberation to the signal
def add_reverberation(signal, num_reflections=REVERB_REFLECTIONS, decay=REVERB_DECAY, delay=REVERB_DELAY):
    reverb_signal = signal.copy()
    for i in range(1, num_reflections + 1):
        delay_samples = int(delay * i * SAMPLE_RATE)
        reflection = (decay ** i) * np.roll(signal, delay_samples)
        reverb_signal += reflection
    return reverb_signal

# Compute TDOA using GCC-PHAT
def gcc_phat(sig1, sig2, fs=SAMPLE_RATE):
    n = len(sig1)
    fft1 = np.fft.fft(sig1, n * 2)
    fft2 = np.fft.fft(sig2, n * 2)
    cross_spectrum = fft1 * np.conj(fft2)
    cross_spectrum /= np.abs(cross_spectrum) + 1e-10
    gcc = np.fft.ifft(cross_spectrum).real
    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n, n)
    tau = lags[np.argmax(gcc)] / fs
    return tau

# Simulate received signals at each microphone with reverberation
def simulate_received_signals(source_direction, mic_positions):
    azimuth, elevation = source_direction
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    original_signal, t = generate_gunshot_signal()
    received_signals = []
    
    for mic_pos in mic_positions:
        delay = np.dot(mic_pos, direction) / SPEED_OF_SOUND
        delay_samples = int(delay * SAMPLE_RATE)
        shifted_signal = np.roll(original_signal, delay_samples)
        reverb_signal = add_reverberation(shifted_signal)
        noisy_signal = add_noise(reverb_signal)
        received_signals.append(noisy_signal)
    
    return received_signals, t, direction

# Estimate DoA using TDOA
def estimate_doa(received_signals, mic_positions):
    tdoas = []
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            tau = gcc_phat(received_signals[i], received_signals[j])
            tdoas.append((i, j, tau))
    
    A = []
    b = []
    for i, j, tau in tdoas:
        mic_diff = mic_positions[i] - mic_positions[j]
        A.append(mic_diff)
        b.append(tau * SPEED_OF_SOUND)
    
    A = np.array(A)
    b = np.array(b)
    direction, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    direction /= np.linalg.norm(direction)
    
    azimuth = np.arctan2(direction[1], direction[0])
    elevation = np.arcsin(direction[2])
    return azimuth, elevation, direction

# Visualize the microphone array and DoA
def plot_doa(mic_positions, true_direction, est_direction):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot microphone positions
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', label='Microphones', s=100)
    
    # Plot true direction
    scale = 0.15  # Scale for visibility
    ax.quiver(0, 0, 0, true_direction[0] * scale, true_direction[1] * scale, true_direction[2] * scale, color='g', label='True Direction')
    
    # Plot estimated direction
    ax.quiver(0, 0, 0, est_direction[0] * scale, est_direction[1] * scale, est_direction[2] * scale, color='r', label='Estimated Direction')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Microphone Array and Direction of Arrival')
    ax.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'microphone_array_doa.png'))
    plt.close()

# Plot received signals
def plot_received_signals(received_signals, t):
    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(received_signals):
        plt.plot(t, signal, label=f'Mic {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Received Signals at Each Microphone')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'received_signals.png'))
    plt.close()

def main():
    true_azimuth = np.radians(45)
    true_elevation = np.radians(30)
    source_direction = (true_azimuth, true_elevation)
    
    received_signals, t, true_direction = simulate_received_signals(source_direction, MIC_POSITIONS)
    
    est_azimuth, est_elevation, est_direction = estimate_doa(received_signals, MIC_POSITIONS)
    
    est_azimuth_deg = np.degrees(est_azimuth)
    est_elevation_deg = np.degrees(est_elevation)
    true_azimuth_deg = np.degrees(true_azimuth)
    true_elevation_deg = np.degrees(true_elevation)
    
    print(f"True Azimuth: {true_azimuth_deg:.2f} degrees")
    print(f"True Elevation: {true_elevation_deg:.2f} degrees")
    print(f"Estimated Azimuth: {est_azimuth_deg:.2f} degrees")
    print(f"Estimated Elevation: {est_elevation_deg:.2f} degrees")
    print(f"Azimuth Error: {abs(est_azimuth_deg - true_azimuth_deg):.2f} degrees")
    print(f"Elevation Error: {abs(est_elevation_deg - true_elevation_deg):.2f} degrees")
    
    # Visualize results
    plot_doa(MIC_POSITIONS, true_direction, est_direction)
    plot_received_signals(received_signals, t)
    
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if __name__ == "__main__":
    main()
