#include <metal_stdlib>
using namespace metal;

static float2 quant_range(threadgroup float2 *ranges, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) {
            ranges[tid] = float2(min(ranges[tid].x, ranges[tid + step].x),
                                 max(ranges[tid].y, ranges[tid + step].y));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return ranges[0];
}

kernel void dynamic_quant_range(
    device const float *input [[buffer(0)]],
    device float2 *partials [[buffer(1)]],
    constant uint2 &size [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 ranges[256];
    float2 range = float2(0.0f);
    for (ulong i = group * 256 + tid; i < size.x; i += size.y * 256) {
        const float value = input[i];
        if (value < range.x) range.x = value;
        if (value > range.y) range.y = value;
    }
    ranges[tid] = range;
    const float2 result = quant_range(ranges, tid);
    if (tid == 0) partials[group] = result;
}

kernel void dynamic_quant_parameters(
    device const float2 *partials [[buffer(0)]],
    device float *scale [[buffer(1)]],
    device uchar *zero [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 ranges[256];
    float2 range = float2(0.0f);
    for (uint i = tid; i < count; i += 256) {
        range = float2(min(range.x, partials[i].x), max(range.y, partials[i].y));
    }
    ranges[tid] = range;
    const float2 result = quant_range(ranges, tid);
    if (tid == 0) {
        const float step = result.x == result.y ? 1.0f : (result.y - result.x) / 255.0f;
        scale[0] = step;
        zero[0] = uchar(min(max(rint(-result.x / step), 0.0f), 255.0f));
    }
}

kernel void dynamic_quant_u8(
    device const float *input [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const uchar *zero [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) {
        const float value = rint(input[i] / scale[0]) + float(zero[0]);
        output[i] = uchar(min(max(value, 0.0f), 255.0f));
    }
}
