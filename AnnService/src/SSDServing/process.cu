#include <iostream>
#include <vector>

template <typename T>
__device__ float ComputeL2Distance(const T *pX, const T *pY, int32_t length)
{
    const T *pEnd4 = pX + ((length >> 2) << 2);
    const T *pEnd1 = pX + length;

    float diff = 0;

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
    }
    while (pX < pEnd1)
    {
        float c1 = ((float)(*pX++) - (float)(*pY++));
        diff += c1 * c1;
    }
    return diff;
}

template <typename T>
__global__ void Process(T *data, T *target, float *distance, uint64_t *offset, int *st, int32_t length, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && st[idx])
    {
        float res = ComputeL2Distance(reinterpret_cast<T *>(data + offset[idx]), reinterpret_cast<T *>(target), length);
        distance[idx] = res;
    }
}

void solve(void *h_target, void *h_data, std::vector<uint64_t> h_offset, std::vector<int> h_st,
           std::vector<float> &h_distance, int64_t bytes, int count, int32_t length, const std::type_info &type, int sizetype)
{
    void *d_target, *d_data;
    uint64_t *d_offset;
    int *d_st;
    float *d_distance;
    cudaMalloc((void **)&d_target, sizetype * length);
    cudaMalloc((void **)&d_data, sizeof(char) * bytes);
    cudaMalloc((void **)d_offset, sizeof(uint64_t) * count);
    cudaMalloc((void **)&d_st, sizeof(int) * count);
    cudaMalloc((void **)d_distance, sizeof(float) * count);
    cudaMemcpy(d_target, h_target, sizetype * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data, sizeof(char) * bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset.data(), sizeof(uint64_t) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_st, h_st.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
    int block = 128;
    int grid = (count + block - 1) / block;
    if (type == typeid(int8_t))
        Process<<<grid, block>>>(reinterpret_cast<int8_t *>(d_data), reinterpret_cast<int8_t *>(d_target), d_distance, d_offset, d_st, length, count);
    else if (type == typeid(uint8_t))
        Process<<<grid, block>>>(reinterpret_cast<uint8_t *>(d_data), reinterpret_cast<uint8_t *>(d_target), d_distance, d_offset, d_st, length, count);
    else if (type == typeid(uint16_t))
        Process<<<grid, block>>>(reinterpret_cast<uint16_t *>(d_data), reinterpret_cast<uint16_t *>(d_target), d_distance, d_offset, d_st, length, count);
    else
        Process<<<grid, block>>>(reinterpret_cast<float *>(d_data), reinterpret_cast<float *>(d_target), d_distance, d_offset, d_st, length, count);
    cudaMemcpy(h_distance.data(), d_distance, sizeof(float) * count, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_target);
    cudaFree(d_offset);
    cudaFree(d_st);
    cudaFree(d_distance);
}