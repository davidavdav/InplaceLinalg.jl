do_gemm!(α, TA, A, TB, B, β, C) = BLAS.gemm!(TA, TB, α, A, B, β, C)      # C ← αAB + βC  (with transpositions as specified)
do_gemm!(α, TA, A::Symmetric, TB, B, β, C) = BLAS.symm!('L', A.uplo, α, blasnode(A), B, β, C)
do_gemm!(α, TA, A, TB, B::Symmetric, β, C) = BLAS.symm!('R', B.uplo, α, blasnode(B), A, β, C)

tr2blas(X::BlasArray) = 'N', X
tr2blas(X::BlasTranspose) = 'T', parent(X)
tr2blas(X::BlasAdjoint) = 'C', parent(X)

function gemm_αABβC!(α, A, B, β, C::BlasMatrix{T}) where T
    α, β = convert.(T, (α, β))
    try do_gemm!(α, tr2blas(A)..., tr2blas(B)..., β, C) catch err
        ip_error(err)
    end
end


C_AB!(C::BlasMatrix{T}, β::Number, α::Number, A::BlasMatrix{T}, B::BlasMatrix{T}) where T = gemm_αABβC!(α, A, B, β, C)
C_AB!(C::BlasMatrix{T}, β::Number, A::BlasMatrix{T}, α::Number, B::BlasMatrix{T}) where T = gemm_αABβC!(α, A, B, β, C)
C_AB!(C::BlasMatrix{T}, β::Number, A::BlasMatrix{T}, B::BlasMatrix{T}, α::Number) where T = gemm_αABβC!(α, A, B, β, C)
