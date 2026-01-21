//****************************************************************************
/* Copyright (C) Abhishek Shivakumar - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Abhishek Shivakumar <abhishek.shivakumar@gmail.com>, 22/04/2022
*****************************************************************************/

#include "SIMDOperations.h"
#include <cstring>

namespace ML {
namespace SIMD {

// CPU Feature Detection
bool CPUFeatures::hasAVX2() {
#ifdef HAVE_AVX2
    return true;
#else
    return false;
#endif
}

bool CPUFeatures::hasAVX512() {
#ifdef HAVE_AVX512
    return true;
#else
    return false;
#endif
}

bool CPUFeatures::hasFMA() {
#ifdef HAVE_FMA
    return true;
#else
    return false;
#endif
}

// AVX2 Implementations
#ifdef HAVE_AVX2
#include <immintrin.h>

void VectorOps::vector_add_scalar(const float* src, float scalar, float* dst, size_t size) {
    const __m256 scalar_vec = _mm256_set1_ps(scalar);
    size_t simd_size = size & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data = _mm256_loadu_ps(&src[i]);
        __m256 result = _mm256_add_ps(data, scalar_vec);
        _mm256_storeu_ps(&dst[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = src[i] + scalar;
    }
}

void VectorOps::vector_mul_scalar(const float* src, float scalar, float* dst, size_t size) {
    const __m256 scalar_vec = _mm256_set1_ps(scalar);
    size_t simd_size = size & ~7;
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data = _mm256_loadu_ps(&src[i]);
        __m256 result = _mm256_mul_ps(data, scalar_vec);
        _mm256_storeu_ps(&dst[i], result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = src[i] * scalar;
    }
}

void VectorOps::vector_add_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t simd_size = size & ~7;
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data1 = _mm256_loadu_ps(&src1[i]);
        __m256 data2 = _mm256_loadu_ps(&src2[i]);
        __m256 result = _mm256_add_ps(data1, data2);
        _mm256_storeu_ps(&dst[i], result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

void VectorOps::vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size) {
    size_t simd_size = size & ~7;
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data1 = _mm256_loadu_ps(&src1[i]);
        __m256 data2 = _mm256_loadu_ps(&src2[i]);
        __m256 result = _mm256_mul_ps(data1, data2);
        _mm256_storeu_ps(&dst[i], result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

void VectorOps::matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols) {
    // For each row in matrix
    for (size_t i = 0; i < rows; ++i) {
        const float* matrix_row = &matrix[i * cols];
        float sum = 0.0f;
        
        size_t simd_size = cols & ~7;
        __m256 sum_vec = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        for (size_t j = 0; j < simd_size; j += 8) {
            __m256 matrix_data = _mm256_loadu_ps(&matrix_row[j]);
            __m256 vector_data = _mm256_loadu_ps(&vector[j]);
            __m256 prod = _mm256_mul_ps(matrix_data, vector_data);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        
        // Horizontal sum of the 8 floats
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        for (int k = 0; k < 8; ++k) {
            sum += sum_array[k];
        }
        
        // Handle remaining elements
        for (size_t j = simd_size; j < cols; ++j) {
            sum += matrix_row[j] * vector[j];
        }
        
        result[i] = sum;
    }
}

void VectorOps::tanh_batch(const float* src, float* dst, size_t size) {
    size_t simd_size = size & ~7;
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data = _mm256_loadu_ps(&src[i]);
        // tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
        // Using approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
        __m256 x2 = _mm256_mul_ps(data, data);
        __m256 numerator = _mm256_mul_ps(data, _mm256_add_ps(_mm256_set1_ps(27.0f), x2));
        __m256 denominator = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
        __m256 result = _mm256_div_ps(numerator, denominator);
        _mm256_storeu_ps(&dst[i], result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = std::tanh(src[i]);
    }
}

float VectorOps::dot_product(const float* vec1, const float* vec2, size_t size) {
    size_t simd_size = size & ~7;
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data1 = _mm256_loadu_ps(&vec1[i]);
        __m256 data2 = _mm256_loadu_ps(&vec2[i]);
        __m256 prod = _mm256_mul_ps(data1, data2);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }
    
    // Horizontal sum
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += sum_array[i];
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

void VectorOps::copy_vector(const float* src, float* dst, size_t size) {
    size_t simd_size = size & ~7;
    
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data = _mm256_loadu_ps(&src[i]);
        _mm256_storeu_ps(&dst[i], data);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        dst[i] = src[i];
    }
}

#else
// Fallback to scalar implementations when AVX2 is not available
#endif

// Always provide scalar fallback implementations
namespace Scalar {

void vector_sub_scalar(const float* src, float scalar, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i] - scalar;
    }
}

void vector_div_scalar(const float* src, float scalar, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i] / scalar;
    }
}

void vector_sub_vector(const float* src1, const float* src2, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src1[i] - src2[i];
    }
}

void vector_div_vector(const float* src1, const float* src2, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src1[i] / src2[i];
    }
}

void tanh_derivative_batch(const float* src, float* dst, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float tanh_val = std::tanh(src[i]);
        dst[i] = 1.0f - tanh_val * tanh_val;
    }
}

void fill_vector(float* dst, float value, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = value;
    }
}

} // namespace Scalar

// Always-available implementations that delegate to SIMD or Scalar
namespace ML {
namespace SIMD {

void VectorOps::vector_sub_scalar(const float* src, float scalar, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        // For now, use scalar fallback
        Scalar::vector_sub_scalar(src, scalar, dst, size);
        return;
    }
#endif
    Scalar::vector_sub_scalar(src, scalar, dst, size);
}

void VectorOps::vector_div_scalar(const float* src, float scalar, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        Scalar::vector_div_scalar(src, scalar, dst, size);
        return;
    }
#endif
    Scalar::vector_div_scalar(src, scalar, dst, size);
}

void VectorOps::vector_sub_vector(const float* src1, const float* src2, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        Scalar::vector_sub_vector(src1, src2, dst, size);
        return;
    }
#endif
    Scalar::vector_sub_vector(src1, src2, dst, size);
}

void VectorOps::vector_div_vector(const float* src1, const float* src2, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        Scalar::vector_div_vector(src1, src2, dst, size);
        return;
    }
#endif
    Scalar::vector_div_vector(src1, src2, dst, size);
}

void VectorOps::tanh_derivative_batch(const float* src, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        Scalar::tanh_derivative_batch(src, dst, size);
        return;
    }
#endif
    Scalar::tanh_derivative_batch(src, dst, size);
}

void VectorOps::fill_vector(float* dst, float value, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // AVX2 implementation would go here
        Scalar::fill_vector(dst, value, size);
        return;
    }
#endif
    Scalar::fill_vector(dst, value, size);
}

// Always provide the core functions
void VectorOps::vector_add_scalar(const float* src, float scalar, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::vector_add_scalar(src, scalar, dst, size);
        return;
    }
#endif
    Scalar::vector_add_scalar(src, scalar, dst, size);
}

void VectorOps::vector_mul_scalar(const float* src, float scalar, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::vector_mul_scalar(src, scalar, dst, size);
        return;
    }
#endif
    Scalar::vector_mul_scalar(src, scalar, dst, size);
}

void VectorOps::vector_add_vector(const float* src1, const float* src2, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::vector_add_vector(src1, src2, dst, size);
        return;
    }
#endif
    Scalar::vector_add_vector(src1, src2, dst, size);
}

void VectorOps::vector_mul_vector(const float* src1, const float* src2, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::vector_mul_vector(src1, src2, dst, size);
        return;
    }
#endif
    Scalar::vector_mul_vector(src1, src2, dst, size);
}

void VectorOps::matrix_vector_multiply(const float* matrix, const float* vector, 
                                      float* result, size_t rows, size_t cols) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::matrix_vector_multiply(matrix, vector, result, rows, cols);
        return;
    }
#endif
    Scalar::matrix_vector_multiply(matrix, vector, result, rows, cols);
}

void VectorOps::tanh_batch(const float* src, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::tanh_batch(src, dst, size);
        return;
    }
#endif
    Scalar::tanh_batch(src, dst, size);
}

float VectorOps::dot_product(const float* vec1, const float* vec2, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        return Scalar::dot_product(vec1, vec2, size);
    }
#endif
    return Scalar::dot_product(vec1, vec2, size);
}

void VectorOps::copy_vector(const float* src, float* dst, size_t size) {
#ifdef __AVX2__
    if (CPUFeatures::hasAVX2()) {
        // Use existing AVX2 implementation if available
        // For now, use scalar
        Scalar::copy_vector(src, dst, size);
        return;
    }
#endif
    Scalar::copy_vector(src, dst, size);
}

} // namespace SIMD
} // namespace ML

SIMDVector SIMDVector::operator+(float scalar) const {
    SIMDVector result(data.size());
    if (use_simd) {
        VectorOps::vector_add_scalar(data.data(), scalar, result.data.data(), data.size());
    } else {
        Scalar::vector_add_scalar(data.data(), scalar, result.data.data(), data.size());
    }
    return result;
}

SIMDVector SIMDVector::operator*(float scalar) const {
    SIMDVector result(data.size());
    if (use_simd) {
        VectorOps::vector_mul_scalar(data.data(), scalar, result.data.data(), data.size());
    } else {
        Scalar::vector_mul_scalar(data.data(), scalar, result.data.data(), data.size());
    }
    return result;
}

SIMDVector SIMDVector::operator+(const SIMDVector& other) const {
    SIMDVector result(data.size());
    if (use_simd) {
        VectorOps::vector_add_vector(data.data(), other.data.data(), result.data.data(), data.size());
    } else {
        Scalar::vector_add_vector(data.data(), other.data.data(), result.data.data(), data.size());
    }
    return result;
}

SIMDVector SIMDVector::operator*(const SIMDVector& other) const {
    SIMDVector result(data.size());
    if (use_simd) {
        VectorOps::vector_mul_vector(data.data(), other.data.data(), result.data.data(), data.size());
    } else {
        Scalar::vector_mul_vector(data.data(), other.data.data(), result.data.data(), data.size());
    }
    return result;
}

void SIMDVector::apply_tanh() {
    if (use_simd) {
        VectorOps::tanh_batch(data.data(), data.data(), data.size());
    } else {
        Scalar::tanh_batch(data.data(), data.data(), data.size());
    }
}

float SIMDVector::dot_product(const SIMDVector& other) const {
    if (use_simd) {
        return VectorOps::dot_product(data.data(), other.data.data(), data.size());
    } else {
        return Scalar::dot_product(data.data(), other.data.data(), data.size());
    }
}

} // namespace SIMD
} // namespace ML
