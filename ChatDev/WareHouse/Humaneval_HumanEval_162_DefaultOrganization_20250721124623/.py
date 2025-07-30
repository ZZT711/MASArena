'''
Given a string 'text', return its md5 hash equivalent string.
If 'text' is an empty string, return None.
>>> string_to_md5('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'
'''
def string_to_md5(text):
    if text == "":
        return None
    return md5_hash(text.encode('utf-8'))
def md5_hash(data):
    # Initialize the MD5 state variables
    a = 0x67452301
    b = 0xEFCDAB89
    c = 0x98BADCFE
    d = 0x10325476
    original_byte_len = len(data)
    original_bit_len = original_byte_len * 8
    data += b'\x80'  # Append the bit '1' to the message
    data += b'\x00' * ((56 - (original_byte_len + 1) % 64) % 64)  # Append zeros
    data += original_bit_len.to_bytes(8, byteorder='little')  # Append original length
    # Process each 512-bit chunk
    for i in range(0, len(data), 64):
        chunk = data[i:i + 64]
        words = [0] * 16
        for j in range(16):
            words[j] = int.from_bytes(chunk[j * 4:j * 4 + 4], byteorder='little')
        # Initialize working variables
        A, B, C, D = a, b, c, d
        # Main loop
        for j in range(64):
            if j < 16:
                f = (B & C) | (~B & D)
                g = j
            elif j < 32:
                f = (D & B) | (~D & C)
                g = (5 * j + 1) % 16
            elif j < 48:
                f = B ^ C ^ D
                g = (3 * j + 5) % 16
            else:
                f = C ^ (B | ~D)
                g = (7 * j) % 16
            f = (f + A + K(j) + words[g]) & 0xFFFFFFFF
            A, D, C, B = D, (B + left_rotate(f, S(j))) & 0xFFFFFFFF, B, C
        # Add the compressed chunk to the current hash value
        a = (a + A) & 0xFFFFFFFF
        b = (b + B) & 0xFFFFFFFF
        c = (c + C) & 0xFFFFFFFF
        d = (d + D) & 0xFFFFFFFF
    # Produce the final hash value
    return to_hex_string(a.to_bytes(4, 'little') + b.to_bytes(4, 'little') +
                         c.to_bytes(4, 'little') + d.to_bytes(4, 'little'))
def K(j):
    # Constants for MD5
    return [0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
            0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
            0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
            0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
            0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
            0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
            0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
            0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
            0xFFFA3942, 0x8771F681, 0x6CA6351B, 0xF57C0FAF,
            0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147,
            0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138,
            0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70, 0x3DA88FC2,
            0x7C9B8D8A, 0xA4506EBB, 0xB5C0FBCF, 0xC67178F2][j]
def left_rotate(x, c):
    return ((x << c) | (x >> (32 - c))) & 0xFFFFFFFF
def to_hex_string(byte_array):
    return ''.join(f'{byte:02x}' for byte in byte_array)