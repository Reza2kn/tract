#include <metal_stdlib>
using namespace metal;

static int integer_zero(device const uchar *data, uint offset, uint type) {
    if (type == 0) return int(((device const char *)data)[offset]);
    if (type == 1) return int(data[offset]);
    return ((device const int *)data)[offset];
}

template<typename A, typename B>
static void integer_matmul(
    device const A *a, device const B *b,
    device const uchar *az, device const uchar *bz,
    device int *output, constant uint *p, uint i) {
    if (i >= p[0]) return;
    const uint rank = p[1];
    uint offsets[4] = {0, 0, 0, 0};
    uint remaining = i;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        const uint coordinate = remaining % p[7 + axis];
        remaining /= p[7 + axis];
        for (uint slot = 0; slot < 4; ++slot) {
            offsets[slot] += coordinate * p[7 + rank * (slot + 1) + axis];
        }
    }
    const int zero_a = integer_zero(az, offsets[2], p[5]);
    const int zero_b = integer_zero(bz, offsets[3], p[6]);
    uint sum = 0;
    for (uint k = 0; k < p[2]; ++k) {
        const int left = int(a[offsets[0] + k * p[3]]) - zero_a;
        const int right = int(b[offsets[1] + k * p[4]]) - zero_b;
        sum += uint(left * right);
    }
    output[i] = as_type<int>(sum);
}

#define INTEGER_MM(NAME, A, B) \
kernel void NAME( \
    device const A *a [[buffer(0)]], device const B *b [[buffer(1)]], \
    device const uchar *az [[buffer(2)]], device const uchar *bz [[buffer(3)]], \
    device int *output [[buffer(4)]], constant uint *parameters [[buffer(5)]], \
    uint i [[thread_position_in_grid]]) { \
    integer_matmul(a, b, az, bz, output, parameters, i); \
}

INTEGER_MM(integer_mm_i8_i8, char, char)
INTEGER_MM(integer_mm_i8_u8, char, uchar)
INTEGER_MM(integer_mm_u8_i8, uchar, char)
INTEGER_MM(integer_mm_u8_u8, uchar, uchar)
