using InplaceLinalg
using Test, LinearAlgebra

@test @macroexpand(@inplace C = A * B)             == :(InplaceLinalg.C_AB!(C, 0, 1, A, B))
@test @macroexpand(@inplace C = α * A * B)         == :(InplaceLinalg.C_AB!(C, 0, α, A, B))
#@test @macroexpand(@inplace C = A * 7B)            == :(InplaceLinalg.C_AB!(C, 0, 7, A, B))
#@test @macroexpand(@inplace C = 3A * B)            == :(InplaceLinalg.C_AB!(C, 0, 3, A, B))
#@test @macroexpand(@inplace C = 3A * 7B)           == :(InplaceLinalg.C_AB!(C, 0, 3 * 7, A, B))

@test @macroexpand(@inplace C = C + α * A * B)     == :(InplaceLinalg.C_AB!(C, 1, α, A, B))
@test @macroexpand(@inplace C = β * C + α * A * B) == :(InplaceLinalg.C_AB!(C, β, α, A, B))
@test @macroexpand(@inplace C = β * C + 2π * A * B) == :(InplaceLinalg.C_AB!(C, β, 2π, A, B))
@test @macroexpand(@inplace C = β * C + 2 * π * A * B) == :(InplaceLinalg.C_AB!(C, β, 2π, A, B))
@test @macroexpand(@inplace C = β * C + A * B)     == :(InplaceLinalg.C_AB!(C, β, 1, A, B))
@test @macroexpand(@inplace C = 2π * C + A * B)    == :(InplaceLinalg.C_AB!(C, 2π, 1, A, B))
@test @macroexpand(@inplace C = 2*π * C + A * B)   == :(InplaceLinalg.C_AB!(C, 2π, 1, A, B))

@test @macroexpand(@inplace C += A * B)            == :(InplaceLinalg.C_AB!(C, 1, 1, A, B))
@test @macroexpand(@inplace C += α * A * B)        == :(InplaceLinalg.C_AB!(C, 1, α, A, B))
@test @macroexpand(@inplace C -= A * B)            == :(InplaceLinalg.C_AB!(C, 1, -1, A, B))
@test @macroexpand(@inplace C -= α * A * B)        == :(InplaceLinalg.C_AB!(C, 1, -α, A, B))

@test @macroexpand(@inplace C += 2C + A * B)       == :(InplaceLinalg.C_AB!(C, 3, 1, A, B))
@test @macroexpand(@inplace C -= 2C + A * B)       == :(InplaceLinalg.C_AB!(C, -1, -1, A, B))
@test @macroexpand(@inplace C += 2.5C + π * A * B) == :(InplaceLinalg.C_AB!(C, 3.5, π, A, B))
@test @macroexpand(@inplace C -= 2.5C + 2exp(1) * A * B) == :(InplaceLinalg.C_AB!(C, -1.5, -(2 * exp(1)), A, B))

@test @macroexpand(@inplace C += A)                == :(InplaceLinalg.C_AB!(C, 1, 1, 1, A))
@test @macroexpand(@inplace C = C + A)             == :(InplaceLinalg.C_AB!(C, 1, 1, 1, A))
@test @macroexpand(@inplace C = 0.1C + 0.2A)       == :(InplaceLinalg.C_AB!(C, 0.1, 1, 0.2, A))

@test @macroexpand(@inplace B *= A)                == :(InplaceLinalg.mult_update!(B, A, Val(1)))

@test @macroexpand(@inplace B /= A)                == :(InplaceLinalg.div_update!(B, /, A))

@test @macroexpand(@inplace B = B / A)             == :(InplaceLinalg.div_update!(B, /, A))
@test @macroexpand(@inplace B = α * B / A)         == :(InplaceLinalg.div_update!(B, α, /, A))
@test @macroexpand(@inplace B = 2π * B / A)        == :(InplaceLinalg.div_update!(B, 2π, /, A))
@test @macroexpand(@inplace B = (2*π) * B / A)     == :(InplaceLinalg.div_update!(B, 2π, /, A))
@test @macroexpand(@inplace B = A \ B)             == :(InplaceLinalg.div_update!(B, \, A))
@test @macroexpand(@inplace B = A \ 2B)            == :(InplaceLinalg.div_update!(B, 2, \, A))
@test @macroexpand(@inplace B = A \ (α * B))       == :(InplaceLinalg.div_update!(B, α, \, A))
@test @macroexpand(@inplace B = A \ (2π * B))      == :(InplaceLinalg.div_update!(B, 2π, \, A))

@test @macroexpand(@inplace B = B * a * b)         == :(InplaceLinalg.mult_update!(B, a, b, Val(1)))
@test @macroexpand(@inplace B = a * B * b)         == :(InplaceLinalg.mult_update!(B, a, b, Val(2)))
@test @macroexpand(@inplace B = a * b * B)         == :(InplaceLinalg.mult_update!(B, a, b, Val(3)))
@test @macroexpand(@inplace B = B * a)             == :(InplaceLinalg.mult_update!(B, a, Val(1)))
@test @macroexpand(@inplace B = a * B)             == :(InplaceLinalg.mult_update!(B, a, Val(2)))

## esoteric cases
@test @macroexpand(@inplace B = -B*A)              == :(InplaceLinalg.mult_update!(B, -1, A, Val(2)))
@test @macroexpand(@inplace B = -A*B)              == :(InplaceLinalg.mult_update!(B, -1, A, Val(3)))

#@test_throws InplaceException try @eval @macroexpand(@inplace C += A \ B) catch err; throw(err.error) end
#@test_throws InplaceException try @eval @inplace(C += B / A) catch err; throw(err.error) end

#@test @macroexpand(@inplace C = B / A)             == :((InplaceLinalg.ip_error)("LHS must be equal to numerator in updating divide"))
#@test @macroexpand(@inplace C = 2B / A)            == :(InplaceLinalg.C_div!(C, 2, B, $(/), A))

A = randn(100, 200)
B = randn(200, 100)

C = zeros(100, 100)
C1 = A * B

## Basic matrix multiplication, gemm!()
@time @inplace C = A * B
@test C ≈ C1


@time @inplace C += A * B
@test C ≈ 2C1

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
@test C ≈ C1

@time @inplace C += At' * B
@test C ≈ 2C1

@time @inplace C = A * Bt'
@test C ≈ C1

@time @inplace C += A * Bt'
@test C ≈ 2C1

@time @inplace C = At' * Bt'
@test C ≈ C1

@time @inplace C += At' * Bt'
@test C ≈ 2C1

## scaling scal!()
#@time @inplace C *= 2
#@test C ≈ 4C1

#@time @inplace C *= π
#@test C ≈ 4π * C1

## copy blascopy!()
#@inplace C *= 0
#@time @inplace C = C1
#@test C == C1

## simple in-place add sxpy!()
#@time @inplace C += C1
#@test C == 2C1

# TRSM =========================================================================
m,n = 10,20
B0 = randn(m,n)
AL = LowerTriangular(randn(m,m))
AR = UpperTriangular(randn(n,n))
AL1 = UnitLowerTriangular(randn(m,m))
AR1 = UnitUpperTriangular(randn(n,n))
DL = Diagonal(randn(m))
DR = Diagonal(randn(n))
rI = UniformScaling(randn())
C = similar(B0)
α = randn()

# test basics: \,/; two different inplace behaviours; Upper and Lower triangles
#B = copy(B0); @inplace C = AL \ B
#@test B == B0
#@test C ≈ AL \ B0

B = copy(B0); @inplace B = AL \ B
@test B ≈ AL \ B0

#B = copy(B0); @inplace C = B / AR
#@test B == B0
#@test C ≈ B0 / AR

B = copy(B0); @inplace B = B / AR
@test B ≈ B0 / AR

#test scaling
B = copy(B0); @inplace B = AL \ (α*B)
@test B ≈ AL \ (α*B0)

B = copy(B0); @inplace B = AL \ (B*α)
@test B ≈ AL \ (α*B0)

B = copy(B0); @inplace B = AL \ 2B
@test B ≈ AL \ 2B0

B = copy(B0); @inplace B = AL \ (2*B)
@test B ≈ AL \ 2B0

B = copy(B0); @inplace B = α*B / AR
@test B ≈ α*B0 / AR

B = copy(B0); @inplace B = B*α / AR
@test B ≈ α*B0 / AR

B = copy(B0); @inplace B = 2B / AR
@test B ≈ 2B0 / AR

B = copy(B0); @inplace B = 2*B / AR
@test B ≈ 2B0 / AR

#test unit-diagonal triangles and UniformScaling 
B = copy(B0); @inplace B = AL1 \ B
@test B ≈ AL1 \ B0

#B = copy(B0); @inplace B = rI \ B
#@test B ≈ rI \ B0

B = copy(B0); @inplace B = B / AR1
@test B ≈ B0 / AR1

#B = copy(B0); @inplace B = B / rI 
#@test B ≈ B0 / rI


#test Diagonal 
#B = copy(B0); @inplace B = DL \ B
#@test B ≈ DL \ B0

#B = copy(B0); @inplace B = B / DR
#@test B ≈ B0 / DR

#Diagonal solve with prescaling not allowed
B = copy(B0); 
@test_throws InplaceException @inplace B = DL \ 2B
@test_throws InplaceException @inplace B = 2B / DR



#test /=
B = copy(B0); @inplace B /= AR
@test B ≈ B0 / AR



#test some disallowed stuff
B = copy(B0)

@test_throws InplaceException @inplace C += AL \ B 
@test_throws InplaceException @inplace C = C + AL \ B 
@test_throws InplaceException @inplace C += B / AR 
@test_throws InplaceException @inplace C = C + B / AR 

Am = randn(m,m)
An = randn(n,n)
@test_throws InplaceException @inplace B = Am*B
@test_throws InplaceException @inplace B = B*An 
@test_throws InplaceException @inplace C = AL / B 

@test_throws InplaceException @inplace C = A*B + C*D 


# TRSV =========================================================================
m = 10
B0 = randn(m)
AL = LowerTriangular(randn(m,m))
AU = UpperTriangular(randn(m,m))
AL1 = UnitLowerTriangular(randn(m,m))
AU1 = UnitUpperTriangular(randn(m,m))
rI = UniformScaling(randn())
C = similar(B0)
α = randn()

# test basics: two different inplace behaviours; triangle variants
#B = copy(B0); @inplace C = AL \ B
#@test B == B0
#@test C ≈ AL \ B0

#B = copy(B0); @inplace B = AL \ B
#@test B ≈ AL \ B0

#B = copy(B0); @inplace B = AU \ B
#@test B ≈ AU \ B0

#B = copy(B0); @inplace B = AL1 \ B
#@test B ≈ AL1 \ B0

#B = copy(B0); @inplace B = AU1 \ B
#@test B ≈ AU1 \ B0

#B = copy(B0); @inplace B = rI \ B
#@test B ≈ rI \ B0


#scaling is disallowed
#@test_throws InplaceException @inplace B = AL \ α*B
#@test_throws InplaceException @inplace B = AL \ B*α
#@test_throws InplaceException @inplace B = AL \ 2B
#@test_throws InplaceException @inplace B = AL \ 2*B

# this too ...
#@test_throws InplaceException @inplace B = B \ AL




