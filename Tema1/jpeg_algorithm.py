# # Sarcini
# 
# 1. [6p] Completați algoritmul JPEG incluzând toate blocurile din imagine.

# 2. [4p] Extindeți la imagini color (incluzând transformarea din RGB în Y'CbCr). Exemplificați pe `scipy.misc.face` folosită în tema anterioară.
# 
# 3. [6p] Extindeți algoritmul pentru compresia imaginii până la un prag MSE impus de utilizator.
# 
# 4. [4p] Extindeți algoritmul pentru compresie video. Demonstrați pe un clip scurt din care luați fiecare cadru și îl tratați ca pe o imagine.


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


class JPEGCompressor:
    def __init__(self, Q_matrix, factor=1):
        self.Q_matrix = factor * Q_matrix

    def apply_dct_quantization(self, block):
        block_dct = dctn(block, type=2)
        block_quantized = np.round(block_dct / self.Q_matrix).astype(np.int32)
        return block_quantized

    def compress_grayscale_image(self, image):
        height, width = image.shape[0], image.shape[1]
        compressed_image = np.zeros((height // block_size, width // block_size), dtype=object)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = image[i:i + block_size, j:j + block_size]
                block_quantized = self.apply_dct_quantization(block)
                block = zlib.compress(block_quantized.flatten())
                compressed_image[i // block_size, j // block_size] = block

        return compressed_image

    def compress_rgb_image(self, image):
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)

        # Compress  each channel separately
        compressed_channels = [self.compress_grayscale_image(yuv[:, :, i]) for i in range(3)]

        return compressed_channels


class JPEGDecompressor:
    def __init__(self, Q_matrix, factor=1):
        self.Q_matrix = factor * Q_matrix

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

        return decompressed_image

    def decompress_rgb_image(self, compressed_image):

        decompressed_channels = [self.decompress_grayscale_image(compressed_channel) for compressed_channel in
                                 compressed_image]

        # Stack the decompressed channels back together
        decompressed_yuv = np.uint8(np.stack(decompressed_channels, axis=-1))

        # Convert the decompressed YCrCb back to RGB
        decompressed_rgb = (cv2.cvtColor(decompressed_yuv, cv2.COLOR_YCR_CB2RGB)).astype(np.int32)

        return decompressed_rgb


def get_mse_err(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse


# 1
X = misc.ascent()

jpeg_compressor = JPEGCompressor(Q_jpeg)
jpeg_decompressor = JPEGDecompressor(Q_jpeg)

compressed = jpeg_compressor.compress_grayscale_image(X)
decompressed = jpeg_decompressor.decompress_grayscale_image(compressed)

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

jpeg_compressor = JPEGCompressor(Q_jpeg)
jpeg_decompressor = JPEGDecompressor(Q_jpeg)

compressed = jpeg_compressor.compress_rgb_image(X)
decompressed = jpeg_decompressor.decompress_rgb_image(compressed)

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
jpeg_compressor = JPEGCompressor(Q_jpeg, factor)
jpeg_decompressor = JPEGDecompressor(Q_jpeg, factor)

compressed = jpeg_compressor.compress_grayscale_image(X)
decompressed = jpeg_decompressor.decompress_grayscale_image(compressed)

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
jpeg_compressor = JPEGCompressor(Q_jpeg, factor)
jpeg_decompressor = JPEGDecompressor(Q_jpeg, factor)

compressed = jpeg_compressor.compress_rgb_image(X)
decompressed = jpeg_decompressor.decompress_rgb_image(compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
fig.suptitle(f"MSE = {get_mse_err(X, decompressed)}", fontsize=16)
axes[0].imshow(X)
axes[0].set_title('Imaginea Originala')

axes[1].imshow(decompressed)
axes[1].set_title('Imagine Decompresata')
fig.tight_layout()
plt.show()
