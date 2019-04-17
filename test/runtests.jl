using InplaceLinalg
using Test

@macroexpand @inplace C = A * B             == :(InplaceLinalg.C_AB!(C, 0, 1, A, B))
@macroexpand @inplace C = α * A * B         == :(InplaceLinalg.C_AB!(C, 0, α, A, B))
@macroexpand @inplace C = A * 7B            == :(InplaceLinalg.C_AB!(C, 0, 7, A, B))
@macroexpand @inplace C = 3A * B            == :(InplaceLinalg.C_AB!(C, 0, 3, A, B))
@macroexpand @inplace C = 3A * 7B           == :(InplaceLinalg.C_AB!(C, 0, 3 * 7, A, B))

@macroexpand @inplace C = C + α * A * B     == :(InplaceLinalg.C_AB!(C, 1, α, A, B))
@macroexpand @inplace C = β * C + α * A * B == :(InplaceLinalg.C_AB!(C, β, α, A, B))
@macroexpand @inplace C = β * C + 2π * A * B == :(InplaceLinalg.C_AB!(C, β, 2π, A, B))
@macroexpand @inplace C = β * C + 2 * π * A * B == :(InplaceLinalg.C_AB!(C, β, 2π, A, B))
@macroexpand @inplace C = β * C + A * B     == :(InplaceLinalg.C_AB!(C, β, 1, A, B))
@macroexpand @inplace C = 2π * C + A * B    == :(InplaceLinalg.C_AB!(C, 2π, 1, A, B))
@macroexpand @inplace C = 2*π * C + A * B   == :(InplaceLinalg.C_AB!(C, 2π, 1, A, B))

@macroexpand @inplace C += A * B            == :(InplaceLinalg.C_AB!(C, 1, 1, A, B))
@macroexpand @inplace C += α * A * B        == :(InplaceLinalg.C_AB!(C, 1, α, A, B))
@macroexpand @inplace C -= A * B            == :(InplaceLinalg.C_AB!(C, 1, -1, A, B))
@macroexpand @inplace C -= α * A * B        == :(InplaceLinalg.C_AB!(C, 1, -α, A, B))

@macroexpand @inplace C += 2C + A * B       == :(InplaceLinalg.C_AB!(C, 3, 1, A, B))
@macroexpand @inplace C -= 2C + A * B       == :(InplaceLinalg.C_AB!(C, -1, -1, A, B))
@macroexpand @inplace C += 2.5C + π * A * B == :(InplaceLinalg.C_AB!(C, 3.5, π, A, B))
@macroexpand @inplace C -= 2.5C + 2exp(1) * A * B == :(InplaceLinalg.C_AB!(C, -1.5, -(2 * exp(1)), A, B))

@macroexpand @inplace C = B / A             == :(InplaceLinalg.C_div(C, 1, B, /, A))
@macroexpand @inplace C = 2B / A            == :(InplaceLinalg.C_div(C, 2, B, /, A))
@macroexpand @inplace C = α * B / A         == :(InplaceLinalg.C_div(C, α, B, /, A))
@macroexpand @inplace C = 2π * B / A        == :(InplaceLinalg.C_div(C, 2π, B, /, A))
@macroexpand @inplace C = 2*π * B / A       == :(InplaceLinalg.C_div(C, 2π, B, /, A))
@macroexpand @inplace C = A \ B             == :(InplaceLinalg.C_div(C, 1, B, \, A))
@macroexpand @inplace C = A \ 2B            == :(InplaceLinalg.C_div(C, 2, B, \, A))
@macroexpand @inplace C = A \ α * B         == :(InplaceLinalg.C_div(C, α, B, \, A))
@macroexpand @inplace C = A \ 2π * B        == :(InplaceLinalg.C_div(C, 2π, B, \, A))
## can't do @macroexpand @inplace B = A \ 2 * π * B yet...

A = randn(100, 200)
B = randn(200, 100)

C = zeros(100, 100)
C1 = A * B

## Basic matrix multiplication, gemm!()
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

if false

## scaling scal!()
@time @inplace C *= 2
@test C ≈ 4C1

@time @inplace C *= π
@test C ≈ 4π * C1

## copy blascopy!()
@inplace C *= 0
@time @inplace C = C1
@test C == C1

## simple in-place add sxpy!()
@time @inplace C += C1
@test C == 2C1

end
