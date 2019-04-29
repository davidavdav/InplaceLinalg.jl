import LinearAlgebra: ldiv!, rdiv!
ldiv!(α::Number, B::BlasArray) = lmul!(1/α,B)
rdiv!(B::BlasArray,α::Number) = rmul!(B,1/α)


# TRSM ==========================================================================
do_trsm!(α, B, side, A::SimpleTriangular)  = BLAS.trsm!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)
do_trsm!(α, B, side, A::UniformScaling) = rmul!(B,α/A.λ)
#do_trsm!(B, α, A::InverseTriangular) = BLAS.trmm!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)

function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(/), A::BlasTriangular{T}) where T
    α = convert(T,α)
    C===B || copyto!(C,B)
    do_trsm!(α, C, 'R', A)
end
function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(\), A::BlasTriangular{T}) where T
    α = convert(T,α)
    C===B || copyto!(C,B)
    do_trsm!(α, C, 'L', A)
end

function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(/), A) where T
    α==1 || ip_error("numerator scaling not available for division of these types (α==1 required).") 
    try
        C===B || copyto!(C,B)
        rdiv!(C,A)
    catch err
        ip_error(err)
    end
end
function C_αBdivA!(C::BlasMatrix{T}, α, B, ::typeof(\), A) where T
    α==1 || ip_error("numerator scaling not available for division of these types (α==1 required).") 
    try
        C===B || copyto!(C,B)
        ldiv!(A,C)
    catch err
        ip_error(err)
    end
end



C_div!(C::BlasMatrix{T}, α::Number, B::BlasMatrix{T}, div::Function, A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, div, A)
#
C_div!(C::BlasMatrix{T}, B::BlasMatrix{T}, α::Number, div::Function, A::BlasTriangular{T}) where T = 
    C_αBdivA!(C, α, B, div, A)
#

C_div!(C::BlasMatrix{T}, α::Number, B::BlasMatrix{T}, div::Function, A) where T = 
    C_αBdivA!(C, α, B, div, A)
#
C_div!(C::BlasMatrix{T}, B::BlasMatrix{T}, α::Number, div::Function, A) where T = 
    C_αBdivA!(C, α, B, div, A)
#

# TRSV ==========================================================================
do_trsv!(B::BlasVector{T}, A::SimpleTriangular{T}) where T = BLAS.trsv!(A.uplo, A.trans, A.diag, A.blasnode, B)
do_trsv!(B, A::UniformScaling) = rmul!(B,1/A.λ)
do_trsv!(B, A) = ldiv!(A,B)
#do_trsv!(B, α, A::InverseTriangular) = BLAS.trmv!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)

function C_αBdivA!(C, α, B, A) 
    α==1 || ip_error("numerator scaling not available for division of these types (α==1 required).")
    try
        C===B || copyto!(C,B)
        do_trsv!(C, A)
    catch err
        ip_error(err)
    end
end

C_div!(C::BlasVector{T}, α::Number, B::BlasVector{T}, ::typeof(\), A) where T = 
    C_αBdivA!(C, α, B, A)
#
C_div!(C::BlasVector{T}, B::BlasVector{T}, α::Number, ::typeof(\), A) where T = 
    C_αBdivA!(C, α, B, A)
#


