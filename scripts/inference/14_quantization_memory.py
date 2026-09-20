import math

PARAMETER_COUNT = 134_515_008


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def raw_weight_bytes(parameter_count: int, bits_per_parameter: int) -> int:
    return parameter_count * bits_per_parameter // 8


def int4_memory(
    parameter_count: int,
    group_size: int,
    scale_bytes: int,
) -> tuple[int, int, int]:
    weight_bytes = math.ceil(parameter_count * 4 / 8)
    num_groups = math.ceil(parameter_count / group_size)
    scale_memory = num_groups * scale_bytes
    total_memory = weight_bytes + scale_memory

    return weight_bytes, scale_memory, total_memory


def main() -> None:
    print(f"Parameters: {PARAMETER_COUNT:,}")

    formats = {
        "FP32": 32,
        "FP16": 16,
        "INT8": 8,
        "INT4": 4,
    }

    for format_name, bits in formats.items():
        memory = raw_weight_bytes(PARAMETER_COUNT, bits)
        print(f"{format_name}: {bytes_to_mib(memory):.2f} MiB")

    print("\nINT4 with FP16 scales:")

    for group_size in (128, 64, 32):
        weight_bytes, scale_memory, total_memory = int4_memory(
            PARAMETER_COUNT,
            group_size,
            scale_bytes=2,
        )

        effective_bits = total_memory * 8 / PARAMETER_COUNT

        print(
            f"Group {group_size:>3}: "
            f"weights={bytes_to_mib(weight_bytes):.2f} MiB, "
            f"scales={bytes_to_mib(scale_memory):.2f} MiB, "
            f"total={bytes_to_mib(total_memory):.2f} MiB, "
            f"effective={effective_bits:.3f} bits/param"
        )


if __name__ == "__main__":
    main()
