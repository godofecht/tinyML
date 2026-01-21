//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#ifndef SIMD_OPERATIONS_H
#define SIMD_OPERATIONS_H

#include <vector>
#include <algorithm>
#include <cmath>

// Platform-specific SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef HAVE_NEON
#include <arm_neon.h>
#endif

namespace ML {
namespace SIMD {

// SIMD-optimized vector operations for x86 AVX2
class VectorOps {
public:
    // Vector-scalar operations with AVX2
    static void vector_add_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_sub_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_mul_scalar(const float* src, float scalar, float* dst, size_t size);
    static void vector_div_scalar(const float* src, float scalar, float* dst, size_t size);
    
    // Vector-vector operations with AVX2
    static void vector_add_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_sub_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size);
    static void vector_div_vector(const float* src1, const float* src2, float* dst, size_t size);
    
    // Matrix-vector multiplication (optimized for feed-forward)
    static void matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols);
    
    // Batched activation functions
    static void tanh_batch(const float* src, float* dst, size_t size);
    static void tanh_derivative_batch(const float* src, float* dst, size_t size);
    
    // Dot product for gradient calculations
    static float dot_product(const float* vec1, const float* vec2, size_t size);
    
    // Memory operations
    static void copy_vector(const float* src, float* dst, size_t size);
    static void fill_vector(float* dst, float value, size_t size);
};

// Fallback implementations for systems without AVX2
namespace Scalar {
    void vector_add_scalar(const float* src, float scalar, float* dst, size_t size);
    void vector_sub_scalar(const float* src, float scalar, float* dst, size_t size);
    void vector_mul_scalar(const float* src, float scalar, float* dst, size_t size);
    void vector_div_scalar(const float* src, float scalar, float* dst, size_t size);
    void vector_add_vector(const float* src1, const float* src2, float* dst, size_t size);
    void vector_sub_vector(const float* src1, const float* src2, float* dst, size_t size);
    void vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size);
    void vector_div_vector(const float* src1, const float* src2, float* dst, size_t size);
    void matrix_vector_multiply(const float* matrix, const float* vector, 
                               float* result, size_t rows, size_t cols);
    void tanh_batch(const float* src, float* dst, size_t size);
    void tanh_derivative_batch(const float* src, float* dst, size_t size);
    float dot_product(const float* vec1, const float* vec2, size_t size);
    void copy_vector(const float* src, float* dst, size_t size);
    void fill_vector(float* dst, float value, size_t size);
}

// CPU feature detection
class CPUFeatures {
public:
    static bool hasAVX2();
    static bool hasAVX512();
    static bool hasFMA();
};

// High-level vector wrapper that automatically chooses best implementation
class SIMDVector {
private:
    std::vector<float> data;
    bool use_simd;
    
public:
    SIMDVector(size_t size = 0);
    SIMDVector(const std::vector<float>& vec);
    
    // Element access
    float& operator[](size_t index) { return data[index]; }
    const float& operator[](size_t index) const { return data[index]; }
    
    // Vector operations
    SIMDVector operator+(float scalar) const;
    SIMDVector operator-(float scalar) const;
    SIMDVector operator*(float scalar) const;
    SIMDVector operator/(float scalar) const;
    
    SIMDVector operator+(const SIMDVector& other) const;
    SIMDVector operator-(const SIMDVector& other) const;
    SIMDVector operator*(const SIMDVector& other) const;
    SIMDVector operator/(const SIMDVector& other) const;
    
    // Neural network specific operations
    void apply_tanh();
    void apply_tanh_derivative();
    float dot_product(const SIMDVector& other) const;
    
    // Utility
    size_t size() const { return data.size(); }
    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }
    void resize(size_t new_size) { data.resize(new_size); }
    
    // Conversion
    std::vector<float> to_std_vector() const { return data; }
};

} // namespace SIMD
} // namespace ML

#endif // SIMD_OPERATIONS_H
