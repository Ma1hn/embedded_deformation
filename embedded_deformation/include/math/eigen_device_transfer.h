#pragma once

#include "imageProcessor/common_types.hpp"
#include "math/device_mat.h"
#include "math/Quaternion.hpp"

//Transfer the device vector/matrix to Eigen
Matrix3f toEigen(const mat33 &rhs);

Matrix4f toEigen(const mat34 &rhs);

Vector3f toEigen(const float3 &rhs);

Vector4f toEigen(const float4 &rhs);

//For basic device vector
float3 fromEigen(const Vector3f &rhs);

float4 fromEigen(const Vector4f &rhs);

void fromEigen(const Isometry3f& se3, Quaternion& rotation, float3& translation);