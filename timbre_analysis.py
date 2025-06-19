import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# Load audio in any format
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Use native sampling rate
        print(f"Audio loaded successfully: {file_path}")
        return y, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        exit(1)

# Extract core timbral features
def extract_features(y, sr):
    features = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)

    flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = np.mean(flatness)

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr'] = np.mean(zcr)

    print("\nExtracted Features:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")

    return features

# Estimate timbre profile using feature heuristics
def estimate_timbre_percentages(features):
    # Use domain-based value scaling for simplicity
    timbre_profile = {
        "Bright": features["spectral_centroid"] / 5000 * 100,
        "Warm": (1 - features["spectral_bandwidth"] / 5000) * 100,
        "Breathy": features["zcr"] * 1000,
        "Harsh": features["spectral_flatness"] * 100,
        "Nasal": (features["mfcc_mean"] / 50) * 100
    }

    # Normalize values to sum 100%
    total = sum(timbre_profile.values())
    for k in timbre_profile:
        timbre_profile[k] = round((timbre_profile[k] / total) * 100, 2)

    return timbre_profile

# Bar chart for timbre profile
def plot_timbre(timbre_profile):
    labels = list(timbre_profile.keys())
    values = list(timbre_profile.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color='cornflowerblue')
    plt.ylabel('Timbre %')
    plt.title('Estimated Timbre Composition')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    # Replace with your file path (support: .wav, .mp3, etc.)
    file_path = r"D:\Timbral Quality identification\Sample audio3.mp3"

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    y, sr = load_audio(file_path)
    features = extract_features(y, sr)
    timbre_profile = estimate_timbre_percentages(features)

    # Output
    print("\nTimbre Profile (%):")
    for k, v in timbre_profile.items():
        print(f"{k}: {v}%")

    plot_timbre(timbre_profile)