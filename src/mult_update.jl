export mult_update!

# B = αBA
mult_update!(B::BlasMatrix{T}, α::Number, A::SimpleTriangular{T}, ::Val{1}) where T = trmm_αAB!(α,A,B,'R')
mult_update!(B::BlasMatrix{T}, α::Number, A::SimpleTriangular{T}, ::Val{2}) where T = trmm_αAB!(α,A,B,'R')
mult_update!(B::BlasMatrix{T}, A::SimpleTriangular{T}, α::Number, ::Val{1}) where T = trmm_αAB!(α,A,B,'R')

# B = BA
mult_update!(B::BlasArray, A, ::Val{1}) = rmul_AB!(A, B) 

# B = αAB
mult_update!(B::BlasMatrix{T}, α::Number, A::SimpleTriangular{T}, ::Val{3}) where T = trmm_αAB!(α,A,B,'L')
mult_update!(B::BlasMatrix{T}, A::SimpleTriangular{T}, α::Number, ::Val{2}) where T = trmm_αAB!(α,A,B,'L')
mult_update!(B::BlasMatrix{T}, A::SimpleTriangular{T}, α::Number, ::Val{3}) where T = trmm_αAB!(α,A,B,'L')

# B = AB
mult_update!(B::BlasArray, A, ::Val{2}) = lmul_AB!(A, B) 


function trmm_αAB!(α, A, B::BlasMatrix{T}, side) where T
    try
        α = convert(T,α)
        BLAS.trmm!(side, A.uplo, A.trans, A.diag, α, A.blasnode, B)    
    catch err
        ip_error(err)
    end

end

function lmul_AB!(A, B) 
    try
        lmul!(A,B)    
    catch err
        ip_error(err)
    end
end

function rmul_AB!(A, B) 
    try
        rmul!(B,A)    
    catch err
        ip_error(err)
    end
end


mult_update!(B, F1, F2, ::Val) = 
    ip_error("multiplicative update not available for this combination of types.") 
#
mult_update!(B, F1, ::Val) = 
    ip_error("multiplicative update not available for this combination of types.") 
#
