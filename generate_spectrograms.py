import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

# ✅ Set your paths
INPUT_DIR = "/home/sonia/Documents/openSource/RF project/DroneRF_Authentication_Kit/DroneAuthToolkit/AR Drone"
OUTPUT_DIR = "Spectrograms"
FS = 1e6  # Sampling frequency in Hz (adjust if needed)

def generate_spectrogram(iq_data, output_path):
    iq_data = iq_data / np.max(np.abs(iq_data))
    f, t, Sxx = spectrogram(iq_data, fs=FS, nperseg=256)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")
                
                try:
                    data = np.loadtxt(filepath, delimiter=',')
                    iq = data[::2] + 1j * data[1::2]
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
                    continue

                # Path for saving image
                relative_path = Path(root).relative_to(input_dir)
                output_file = os.path.join(output_dir, relative_path, file.replace(".csv", ".png"))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                generate_spectrogram(iq, output_file)

if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
