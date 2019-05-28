export div_update!

"""
    div_upate

Inplace solve (left or right matrix divide). Usage:    

    div_update!(B,α,/,A)    # B = (αB)/A 

    div_update!(B,/,A)      # B = B/A 

    div_update!(B,α,\\,A)    # B = A\\(αB) 

    div_update!(B,\\,A)      # B = A\\B 

Matrix B is updated inplace. α is scalar. A must be triangular or simpler.     
"""
function div_update! 
end



for (div, side, fun) in ( ( :/, 'R', :(rdiv!(B,A)) ), 
                          ( :\, 'L', :(ldiv!(A,B)) ) )
    @eval begin  
        function div_update!(B::BlasMatrix{T}, α::Number, ::typeof($div), A::BlasTriangular{T}) where T
            try
                α = convert(T,α)
                BLAS.trsm!($side, A.uplo, A.trans, A.diag, α, A.blasnode, B)
            catch err
                ip_error(err)
            end
        end

        function div_update!(B::BlasArray, ::typeof($div), A) 
            try
                $fun
            catch err
                ip_error(err)
            end
        end

    end
end


div_update!(B,α,f::Function,A) = ip_error("solve method not available for this combination of types .") 
div_update!(B,f::Function,A) = ip_error("solve method not available for this combination of types .") 

