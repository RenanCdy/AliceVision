#pragma once
#include <sycl/sycl.hpp>

namespace custom_sycl {
struct custom_float3 {
    sycl::marray<float, 3> data;

    custom_float3() = default;

    custom_float3(float x, float y, float z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    // Access operators
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }

     // Assignment operator from sycl::float3
    custom_float3& operator=(const sycl::float3& other) {
        data[0] = other.x();
        data[1] = other.y();
        data[2] = other.z();
        return *this;
    }

    custom_float3 operator-(const sycl::float3& other) const {
        return custom_float3(data[0] - other.x(), data[1] - other.y(), data[2] - other.z());
    }
    custom_float3 operator+(const sycl::float3& other) const {
        return custom_float3(data[0] + other.x(), data[1] + other.y(), data[2] + other.z());
    }
    operator sycl::float3() const {
        return sycl::float3(data[0], data[1], data[2]);
    }
};
};