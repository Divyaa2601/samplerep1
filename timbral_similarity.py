import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load audio
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Extract features for timbral quality
def extract_timbral_features(y, sr):
    features = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(bandwidth)

    flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = np.mean(flatness)

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr'] = np.mean(zcr)

    return features

# Convert features into timbre profile (%)
def estimate_timbre_profile(features):
    timbre_profile = {
        'Bright': features['spectral_centroid'] / 5000 * 100,
        'Warm': (1 - features['spectral_bandwidth'] / 5000) * 100,
        'Breathy': features['zcr'] * 1000,
        'Harsh': features['spectral_flatness'] * 100,
        'Nasal': features['mfcc_mean'] / 50 * 100
    }

    total = sum(timbre_profile.values())
    for k in timbre_profile:
        timbre_profile[k] = round((timbre_profile[k] / total) * 100, 2)

    return timbre_profile

# Compare and visualize timbre profiles
def plot_timbre_profiles(profile1, profile2, file1, file2):
    labels = list(profile1.keys())
    values1 = list(profile1.values())
    values2 = list(profile2.values())
    x = np.arange(len(labels))
    width = 0.35

    # Bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, values1, width, label=os.path.basename(file1), color='skyblue')
    plt.bar(x + width/2, values2, width, label=os.path.basename(file2), color='salmon')
    plt.xticks(x, labels)
    plt.ylabel('Timbre %')
    plt.title('Timbre Quality Comparison')
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Radar chart
    def radar_plot(values1, values2, labels):
        values1 += values1[:1]
        values2 += values2[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values1, 'b-', label=os.path.basename(file1))
        ax.fill(angles, values1, 'b', alpha=0.25)

        ax.plot(angles, values2, 'r-', label=os.path.basename(file2))
        ax.fill(angles, values2, 'r', alpha=0.25)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Timbre Radar Plot")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

    radar_plot(values1.copy(), values2.copy(), labels)

# Cosine similarity of profiles
def compare_profiles(profile1, profile2):
    v1 = list(profile1.values())
    v2 = list(profile2.values())
    sim = cosine_similarity([v1], [v2])[0][0]
    print(f"\nTimbre Profile Cosine Similarity: {sim:.4f}")
    if sim > 0.95:
        print("→ Timbre profiles are very similar.")
    elif sim > 0.80:
        print("→ Timbre profiles are moderately similar.")
    else:
        print("→ Timbre profiles are quite different.")

# Main
if __name__ == "__main__":
    file1 = r"D:\Timbral Quality identification\Sample audio1.mp3"
    file2 = r"D:\Timbral Quality identification\Sample audio2.mp3"

    if not os.path.exists(file1) or not os.path.exists(file2):
        print("One or both files not found.")
        exit()

    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    features1 = extract_timbral_features(y1, sr1)
    features2 = extract_timbral_features(y2, sr2)

    profile1 = estimate_timbre_profile(features1)
    profile2 = estimate_timbre_profile(features2)

    print("\nTimbre Profile - File 1:")
    for k, v in profile1.items():
        print(f"{k}: {v}%")

    print("\nTimbre Profile - File 2:")
    for k, v in profile2.items():
        print(f"{k}: {v}%")

    compare_profiles(profile1, profile2)
    plot_timbre_profiles(profile1, profile2, file1, file2)
