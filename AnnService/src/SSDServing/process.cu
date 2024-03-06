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
__global__ void Process(T *data, T *target, float *distance, int32_t length, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int sz = sizeof(T) * length;
        float res = ComputeL2Distance(reinterpret_cast<T *>(data + idx * sz), reinterpret_cast<T *>(target), length);
        distance[idx] = res;
    }
}

void solve(void *target, void *data, std::vector<float> &h_distance, int count, int length, const std::type_info &type)
{
    float *d_distance;
    cudaMalloc((void **)&d_distance, sizeof(float) * count);
    int block = 64;
    int grid = (count + block - 1) / block;
    if (type == typeid(int8_t))
        Process<<<grid, block>>>(reinterpret_cast<int8_t *>(data), reinterpret_cast<int8_t *>(target), d_distance, length, count);
    else if (type == typeid(uint8_t))
        Process<<<grid, block>>>(reinterpret_cast<uint8_t *>(data), reinterpret_cast<uint8_t *>(target), d_distance, length, count);
    else if (type == typeid(uint16_t))
        Process<<<grid, block>>>(reinterpret_cast<uint16_t *>(data), reinterpret_cast<uint16_t *>(target), d_distance, length, count);
    else
        Process<<<grid, block>>>(reinterpret_cast<float *>(data), reinterpret_cast<float *>(target), d_distance, length, count);
    cudaDeviceSynchronize();
    cudaMemcpy(h_distance.data(), d_distance, sizeof(float) * count, cudaMemcpyDeviceToHost);
    cudaFree(data);
    cudaFree(target);
    cudaFree(d_distance);
}