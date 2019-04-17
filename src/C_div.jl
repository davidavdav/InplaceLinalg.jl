do_trsm!(α, B, side, A::SimpleTriangular)  = BLAS.trsm!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)
do_trsm!(α, B, side, A::UniformScaling) = rmul!(B,α/A.λ)
do_trsm!(α, B, side, A::Number) = rmul!(B,α/A)
#do_trsm!(B, α, A::InverseTriangular) = BLAS.trmm!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)

function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(/), A) where T
    α = convert(T,α)
    C===B || copyto!(C,B)
    do_trsm!(α, C, 'R', A)
end
function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(\), A) where T
    α = convert(T,α)
    C===B || copyto!(C,B)
    do_trsm!(α, C, 'L', A)
end




C_div!(C::BlasMatrix{T}, α::Number, B::BlasMatrix{T}, div::Function, A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, div, A)
#
C_div!(C::BlasMatrix{T}, B::BlasMatrix{T}, α::Number, div::Function, A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, div, A)
#

C_div!(C::BlasMatrix{T}, α::Number, B::BlasMatrix{T}, div::Function, A::Number) where T = 
    C_αBdivA!(C, α, B, div, A)
#
C_div!(C::BlasMatrix{T}, B::BlasMatrix{T}, α::Number, div::Function, A::Number) where T = 
    C_αBdivA!(C, α, B, div, A)
#

