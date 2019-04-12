using InplaceLinalg
using Test

A = randn(10, 20)
B = randn(20, 10)

C = zeros(10, 10)
C1 = A * B
@time @inplace C = A * B

@test C1 == C
