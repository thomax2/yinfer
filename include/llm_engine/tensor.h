#pragma once

#include <vector>
#include <numeric>
#include <cstdlib>
#include <cstring>

namespace llm_engine {

enum class DataType {
    FP32,
    FP16,
    INT8
};

enum class DeviceType {
    CPU,
    CUDA
};

inline size_t dtype_size(DataType t) {
    switch (t) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
        default: return 0;
    }
}

class Tensor {
public:
    std::vector<int> shape;
    std::vector<int> stride;    /*  
                                    在内存中移动到下一个元素需要跳过多少个位置, 例如一个2x3的矩阵，行优先存储，那么shape是{2,3}，
                                    stride是{3,1}，因为移动到下一行需要跳过3个元素，移动到下一列需要跳过1个元素。           
                                */ 

    void* data = nullptr;

    DataType dtype = DataType::FP32;
    DeviceType device = DeviceType::CPU;

public:

    Tensor() = default;

    Tensor( const std::vector<int>& s,
            DataType t = DataType::FP32,
            DeviceType dev = DeviceType::CPU)
        : shape(s), dtype(t), device(dev) {
    
        compute_stride();
        allocate();
    }

    ~Tensor() {
        free_memory();
    }

    size_t size() const {
        // 从头到尾遍历 所有元素 并相乘
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    size_t bytes() const {
        return size() * dtype_size(dtype);
    }

    template<typename T>    // T 是一个占位符，代表任意类型。
    T* ptr() {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    const T* ptr() const {
        return reinterpret_cast<const T*>(data);
    }

private:
    void compute_stride() {
        stride.resize(shape.size());
        int acc = 1;
        for(int i = shape.size() - 1; i >= 0; i--) {
            stride[i] = acc;
            acc *= shape[i];
        }
    }

    void allocate() {
        if (device != DeviceType::CPU)
            // 目前只支持CPU，其他设备暂不支持
            return;

        data = std::malloc(bytes());
        if(data)
            std::memset(data, 0, bytes());
    }

    void free_memory() {
        if (device == DeviceType::CPU && data) {
            std::free(data);
            data = nullptr;
        }
    }
};

} // namespace llm_engine