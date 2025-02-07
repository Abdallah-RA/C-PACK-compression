import numpy as np
import cv2
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt

# Optimized Compression Class
class FastCpackCompression:
    def __init__(self, word_size=32, dict_size=8192):
        self.word_size = word_size
        self.dict_entries = dict_size
        self.dictionary = {}
        self.next_dict_index = 0

    def compress(self, data_in):
        if data_in == 0:
            return 0b00  # Zero Word compression

        if data_in in self.dictionary:
            return (0b10 << 4) | self.dictionary[data_in]  # Full Match compression

        if data_in & 0xFFFFFF00 == 0:
            return (0b1101 << 8) | (data_in & 0xFF)  # 1-Byte Sign Extended Match

        dict_key = data_in & 0xFFFFFF00
        if dict_key in self.dictionary:
            return (0b1110 << 12) | (self.dictionary[dict_key] << 8) | (data_in & 0xFF)  # 3-Byte Partial Match

        dict_key = data_in & 0xFFFF0000
        if dict_key in self.dictionary:
            return (0b1100 << 12) | (self.dictionary[dict_key] << 8) | (data_in & 0xFFFF)  # 2-Byte Partial Match

        if self.next_dict_index < self.dict_entries:
            self.dictionary[data_in] = self.next_dict_index
            self.next_dict_index += 1

        return (0b01 << 32) | data_in  # Uncompressed Word

# Read files efficiently
def read_image_file(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image.flatten()

def read_video_file(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame.flatten())
    cap.release()
    return np.concatenate(frames)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return np.frombuffer(data.encode('utf-8'), dtype=np.uint8)

# Preprocess data for SRAM, ensuring proper padding
def preprocess_data(data, word_size=32):
    bytes_per_word = word_size // 8
    num_words = len(data) // bytes_per_word
    padding_needed = max(0, (num_words + 1) * bytes_per_word - len(data))

    if padding_needed > 0:
        data = np.pad(data, (0, padding_needed), 'constant')

    words = np.frombuffer(data.tobytes(), dtype=np.uint32)
    return words

# Function to compress data chunks in parallel
def compress_chunk(chunk, compressor):
    return [compressor.compress(word) for word in chunk]

# Function to perform the compression test
def fast_compression(data, data_type):
    words = preprocess_data(data)
    compressor = FastCpackCompression()

    chunk_size = max(1, len(words) // 8)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    compressed_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda chunk: compress_chunk(chunk, compressor), chunks)
        for result in results:
            compressed_results.extend(result)

    original_size_bytes = len(words) * 4  # Convert from words to bytes
    compressed_size_bytes = sum((len(bin(c)) - 2) // 8 for c in compressed_results)  # Convert to bytes

    compression_ratio = (1 - (compressed_size_bytes / original_size_bytes)) * 100
    return data_type, original_size_bytes, compressed_size_bytes, compression_ratio

# Generate best-fit large random data patterns for the compression algorithm
def generate_large_patterned_data(size):
    pattern = np.array([0, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000, 0x12345678], dtype=np.uint32)
    data = np.tile(pattern, size // len(pattern))
    np.random.shuffle(data)
    return data

# Define file paths
image_file_path = 'RA.jpg'  # Replace with actual image file path
video_file_path = 'RA.mp4'  # Replace with actual video file path
text_file_path = 'RA.txt'   # Replace with actual text file path

# Run tests
print("Running tests, please wait...")

test_results = []

# Large patterned data test (optimized for compression)
large_patterned_data = generate_large_patterned_data(10**6)
test_results.append(fast_compression(large_patterned_data, "Large Patterned Data"))

# Image file test
image_data = read_image_file(image_file_path)
test_results.append(fast_compression(image_data, "Image File"))

# Text file test
text_data = read_text_file(text_file_path)
test_results.append(fast_compression(text_data, "Text File"))

# Video file test
video_data = read_video_file(video_file_path)
test_results.append(fast_compression(video_data, "Video File"))

# Display results as a table using pandas
df_results = pd.DataFrame(test_results, columns=["Data Type", "Original Size (Bytes)", "Compressed Size (Bytes)", "Compression Ratio (%)"])
print("\nCompression Test Results:")
print(df_results.to_string(index=False))

# Plot results
df_results.plot.bar(x="Data Type", y=["Original Size (Bytes)", "Compressed Size (Bytes)"], figsize=(10, 6), title="Compression Results")
plt.xlabel('Data Type')
plt.ylabel('Size (Bytes)')
plt.grid(True)
plt.show()
