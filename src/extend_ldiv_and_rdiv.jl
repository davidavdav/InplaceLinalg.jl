import LinearAlgebra: ldiv!, rdiv!

ldiv!(α::Number, B::BlasArray) = lmul!(1 / α, B)
rdiv!(B::BlasArray, α::Number) = rmul!(B, 1 /α )

ldiv!(U::UniformScaling, B::BlasArray) = lmul!(1 / U.λ, B)
rdiv!(B::BlasArray, U::UniformScaling) = rmul!(B, 1 / U.λ)
