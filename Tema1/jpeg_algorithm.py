# # Sarcini
# 
# 1. [6p] Completați algoritmul JPEG incluzând toate blocurile din imagine.

# 2. [4p] Extindeți la imagini color (incluzând transformarea din RGB în Y'CbCr). Exemplificați pe `scipy.misc.face` folosită în tema anterioară.
# 
# 3. [6p] Extindeți algoritmul pentru compresia imaginii până la un prag MSE impus de utilizator.
# 
# 4. [4p] Extindeți algoritmul pentru compresie video. Demonstrați pe un clip scurt din care luați fiecare cadru și îl tratați ca pe o imagine.

import pickle
from skimage import io
import zlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
import matplotlib

matplotlib.use('TkAgg')

Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 28, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
block_size = 8


class JPEGAlgorithm:
    def __init__(self, Q_matrix, factor=1):
        self.Q_matrix = factor * Q_matrix
        self.padding = [0, 0]
        self.original_dimensions = (0, 0)
        self.fps = 0
        self.frame_count = 0
        self.frame_width = 0
        self.frame_height = 0

    def apply_dct_quantization(self, block):
        block_dct = dctn(block, type=2)
        block_quantized = np.round(block_dct / self.Q_matrix).astype(np.int32)
        return block_quantized

    def compress_grayscale_image(self, image):
        original_height, original_width = image.shape[0], image.shape[1]

        # Calculate padding
        if original_height % 8 != 0:
            self.padding[0] = block_size - original_height % 8
        if original_width % 8 != 0:
            self.padding[1] = block_size - original_width % 8

        new_image = np.pad(image, ((0, self.padding[0]), (0, self.padding[1])), mode='edge')
        height, width = new_image.shape[0], new_image.shape[1]

        compressed_image = np.zeros((height // block_size, width // block_size), dtype=object)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = new_image[i:i + block_size, j:j + block_size]
                block_quantized = self.apply_dct_quantization(block)
                block = zlib.compress(block_quantized.flatten().tobytes())
                compressed_image[i // block_size, j // block_size] = block
        self.original_dimensions = (original_height, original_width)

        return compressed_image

    def compress_rgb_image(self, image):
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)

        # Compress  each channel separately
        compressed_channels = [self.compress_grayscale_image(yuv[:, :, i]) for i in range(3)]

        return compressed_channels

    def apply_inverse_dct_quantization(self, block_quantized):
        block_dct = block_quantized * self.Q_matrix
        block_idct = idctn(block_dct, type=2)
        return block_idct

    def decompress_grayscale_image(self, compressed_image):
        height, width = compressed_image.shape
        height *= block_size
        width *= block_size
        decompressed_image = np.zeros((height, width), dtype=np.int32)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block_decompressed = zlib.decompress(compressed_image[i // block_size, j // block_size])
                block_decompressed = np.frombuffer(block_decompressed, dtype=np.int32)
                block_decompressed = block_decompressed.reshape(block_size, block_size)
                block_decompressed = self.apply_inverse_dct_quantization(block_decompressed)
                decompressed_image[i:i + block_size, j:j + block_size] = block_decompressed

        decompressed_image = decompressed_image[:self.original_dimensions[0], :self.original_dimensions[1]]
        return decompressed_image

    def decompress_rgb_image(self, compressed_image):

        decompressed_channels = [self.decompress_grayscale_image(compressed_channel) for compressed_channel in
                                 compressed_image]

        # Stack the decompressed channels back together
        decompressed_yuv = np.uint8(np.stack(decompressed_channels, axis=-1))

        # Convert the decompressed YCrCb back to RGB
        decompressed_rgb = (cv2.cvtColor(decompressed_yuv, cv2.COLOR_YCR_CB2RGB)).astype(np.int32)

        return decompressed_rgb

    # Compress and decompress videos
    def compress_video(self, input_path,output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = fps
        self.frame_count = frame_count
        self.frame_width = frame_width
        self.frame_height = frame_height

        print(f"Frames:{frame_count}")
        # wirte in a file
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Compress each frame
            compressed_frame = self.compress_rgb_image(frame)

            # Uncompress each frame
            decompressed_frame = self.decompress_rgb_image(compressed_frame)

            # Write the decompress frame
            out.write(decompressed_frame.astype(np.uint8))

        out.release()
        cap.release()


def get_mse_err(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse


# 1
X = misc.ascent()

jpeg_alg = JPEGAlgorithm(Q_jpeg)

compressed = jpeg_alg.compress_grayscale_image(X)
decompressed = jpeg_alg.decompress_grayscale_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(f"MSE = {get_mse_err(X, decompressed)}", fontsize=16)
axes[0].imshow(X, cmap=plt.cm.gray)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed, cmap=plt.cm.gray)
axes[1].set_title('Imagine Decompresata')
fig.tight_layout()
plt.show()

# 2
X = misc.face()

jpeg_alg = JPEGAlgorithm(Q_jpeg)

compressed = jpeg_alg.compress_rgb_image(X)
decompressed = jpeg_alg.decompress_rgb_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(f"MSE = {get_mse_err(X, decompressed)}", fontsize=16)
axes[0].imshow(X)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed)
axes[1].set_title('Imagine Decompresata')
fig.tight_layout()
plt.show()

# 3
X = misc.ascent()
factor = 10
jpeg_alg = JPEGAlgorithm(Q_jpeg, factor)

compressed = jpeg_alg.compress_grayscale_image(X)
decompressed = jpeg_alg.decompress_grayscale_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(f"MSE = {get_mse_err(X, decompressed)}", fontsize=16)
axes[0].imshow(X, cmap=plt.cm.gray)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed, cmap=plt.cm.gray)
axes[1].set_title('Imagine Decompresata')
fig.tight_layout()
plt.show()

X = misc.face()
factor = 0.5
jpeg_alg = JPEGAlgorithm(Q_jpeg, factor)

compressed = jpeg_alg.compress_rgb_image(X)
decompressed = jpeg_alg.decompress_rgb_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(f"MSE = {get_mse_err(X, decompressed)}", fontsize=16)
axes[0].imshow(X)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed)
axes[1].set_title('Imagine Decompresata')
fig.tight_layout()
plt.show()

# Test padding for images that have dimension not divisible with block_size
image_path = 'bunny.png'
original_image = io.imread(image_path)
original_image = original_image[:, :, :3]
jpeg_alg = JPEGAlgorithm(Q_jpeg)
compressed = jpeg_alg.compress_rgb_image(original_image)
decompressed = jpeg_alg.decompress_rgb_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"MSE = {get_mse_err(original_image, decompressed)}", fontsize=16)
axes[0].imshow(original_image)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed)
axes[1].set_title('Imaginea Decompresata')

fig.tight_layout()
plt.show()

jpeg_alg = JPEGAlgorithm(Q_jpeg, 500)
input_video_path = 'shorter_shorter_video.mp4'
output_video_path = 'output_video_compressed.mp4'

jpeg_alg.compress_video(input_video_path, output_video_path)
