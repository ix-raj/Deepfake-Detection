import pickle
import numpy as np

# Load file
with open('power_spectrum_10000.pkl', 'rb') as f:
    data = pickle.load(f)

with open('power_spectrum_1000.pkl', 'rb') as f:
    data1 = pickle.load(f)


# Print type
print("Type:", type(data))

# Keys inside
print("Keys:", data.keys())

# Shape of data
print("X shape:", data["power_spectrum"].shape)
print("y shape:", data["label"].shape)

# Sample values
print("\nSample X:", data["power_spectrum"][0][:10])
print("Sample y:", data["label"][:10])

print("\nLabel distribution:", np.bincount(data["label"].astype(int)))

print("Type:", type(data1))

# Keys inside
print("Keys:", data1.keys())

# Shape of data
print("X shape:", data1["power_spectrum"].shape)
print("y shape:", data1["label"].shape)

# Sample values
print("\nSample X:", data1["power_spectrum"][0][:10])
print("Sample y:", data1["label"][:10])

print("\nLabel distribution:", np.bincount(data1["label"].astype(int)))