# TRSM ==========================================================================
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

# TRSV ==========================================================================
do_trsv!(B, A::SimpleTriangular)  = BLAS.trsv!(A.uplo, A.trans, A.diag, A.blasnode, B)
do_trsv!(B, A::UniformScaling) = rmul!(B,1/A.λ)
do_trsv!(B, A::Number) = rmul!(B,1/A)
#do_trsv!(B, α, A::InverseTriangular) = BLAS.trmv!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)

function C_αBdivA!(C, α, B, A) 
    α==1 || ip_error("scaling not available for triangular solve, with vector RHS (α==1 required).")
    C===B || copyto!(C,B)
    do_trsv!(C, A)
end

C_div!(C::BlasVector{T}, α::Number, B::BlasVector{T}, ::typeof(\), A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, A)
#
C_div!(C::BlasVector{T}, B::BlasVector{T}, α::Number, ::typeof(\), A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, A)
#

C_div!(C::BlasVector{T}, α::Number, B::BlasVector{T}, ::typeof(\), A::Number) where T = 
    C_αBdivA!(C, 1, B, convert(T, A/α) )
#
C_div!(C::BlasVector{T}, B::BlasVector{T}, α::Number, ::typeof(\), A::Number) where T = 
    C_αBdivA!(C, 1, B, convert(T, A/α) )
#

