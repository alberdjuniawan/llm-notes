def encode_int4(value: int) -> int:
    if value < -8 or value > 7:
        raise ValueError(f"Value {value} is outside signed INT4 range [-8, 7].")

    return value & 0x0F


def decode_int4(value: int) -> int:
    if value & 0x08:
        return value - 16

    return value


def pack_int4_pair(low: int, high: int) -> int:
    low_nibble = encode_int4(low)
    high_nibble = encode_int4(high)

    return low_nibble | (high_nibble << 4)


def unpack_int4_pair(byte: int) -> tuple[int, int]:
    low_nibble = byte & 0x0F
    high_nibble = (byte >> 4) & 0x0F

    return decode_int4(low_nibble), decode_int4(high_nibble)


def main() -> None:
    values = [-7, -3, 1, 1, 1, 7]

    print(f"Original INT4 values: {values}")
    print(f"Number of values: {len(values)}")
    print(f"Theoretical storage: {len(values) * 4 // 8} bytes")

    packed = []

    for index in range(0, len(values), 2):
        byte = pack_int4_pair(values[index], values[index + 1])
        packed.append(byte)

    print(f"Packed bytes: {packed}")
    print(f"Packed hex: {[f'0x{byte:02X}' for byte in packed]}")
    print(f"Actual storage: {len(packed)} bytes")

    unpacked = []

    for byte in packed:
        low, high = unpack_int4_pair(byte)
        unpacked.extend([low, high])

    print(f"Unpacked INT4 values: {unpacked}")
    print(f"Round-trip successful: {unpacked == values}")


if __name__ == "__main__":
    main()
