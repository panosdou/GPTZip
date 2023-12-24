import zlib

def calculate_compression_ratio(data):

    data_bytes = ''.join(data).encode('utf-8')
    # compress = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    # compressed_data = compress.compress(data_bytes)
    # compressed_data += compress.flush()

    compressed_data = zlib.compress(data_bytes, level = 6)
    #print(len(compressed_data))
    #print(compressed_data)
    #print(len(data_bytes))
    file = open('Zlib/compressed.bin', 'wb')
    file.write(compressed_data)
    file.close()

    compression_ratio =   len(compressed_data) * 8  / len(data_bytes) 

    return compression_ratio

with open('Zlib/text.txt', 'r') as file:
    data = file.read()
    file.close()

compression_ratio = calculate_compression_ratio(data)

print(compression_ratio)

#### test for decompression ####

# with open('compressed.bin', 'rb') as file:
#     compressed = file.read()
#     original = zlib.decompress(compressed)
#     message = original.decode('utf-8', errors='ignore')
#     file.close()

# print(message)
