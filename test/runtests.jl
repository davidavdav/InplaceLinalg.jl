using InplaceLinalg
using Test

A = randn(100, 200)
B = randn(200, 100)

C = zeros(100, 100)
C1 = A * B

## Basic matrix multiplication
@time @inplace C = A * B
@test C == C1

@time @inplace C += A * B
@test C == 2C1

## With scaling factor
@time @inplace C = 3.0 * A * B
@test C ≈ 3C1

@time @inplace C += 5 * A * B
@test C ≈ 8C1

@time @inplace C += 2 * 4.0f0 * A * B
@test C ≈ 16C1

## Transpose etc.

Bt = collect(B')
At = collect(A')

@time @inplace C = At' * B
@test C == C1

@time @inplace C += At' * B
@test C == 2C1

@time @inplace C = A * Bt'
@test C == C1

@time @inplace C += A * Bt'
@test C == 2C1

@time @inplace C = At' * Bt'
@test C == C1

@time @inplace C += At' * Bt'
@test C == 2C1

## scaling
@time @inplace C *= 2
@test C ≈ 4C1

@time @inplace C *= π
@test C ≈ 4π * C1