import zlib

def calculate_compression_ratio(data):

    data_bytes = ''.join(data).encode()
    #data_bytes = ''.join(format(ord(x), 'b') for x in data).encode()
    #print(data_bytes)
    compress = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    compressed_data = compress.compress(data_bytes)
    compressed_data += compress.flush()

    # compressed_data = zlib.compress(data_bytes, level = 6)
    #print(data_bytes)

    compression_ratio = len(data_bytes) / len(compressed_data)

    return compression_ratio


#data_letters = ["hello this is a test a little longer than this test would be if i did not write more than this but that's ok and I'm trying to figure out what's wrong with this can you help? Are you sure?"]
#data_numbers = [1, 2, 3, 4, 5, 6, 7, 8]

with open('text.txt', 'r') as file:
    data = file.read()

compression_ratio = calculate_compression_ratio(data)



print(compression_ratio)
