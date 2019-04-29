export div_update!

div_update!(B,α,f::Function,A) = ip_error("inplace solve for this combination of types not implemented.") 

for (div, side, fun) in ( ( :/, 'R', :(rdiv!(B,A)) ), 
                          ( :\, 'L', :(ldiv!(A,B)) ) )
    @eval begin  
        function div_update!(B::BlasMatrix{T}, α::Number, ::typeof($div), A::SimpleTriangular{T}) where T
            try
                α = convert(T,α)
                BLAS.trsm!($side, A.uplo, A.trans, A.diag, α, A.blasnode, B)
            catch err
                ip_error(err)
            end
        end

        function div_update!(B::BlasArray, α::Number, ::typeof($div), A) 
            α==1 || ip_error("numerator scaling not available for division of these types (α==1 required).") 
            try
                $fun
            catch err
                ip_error(err)
            end
        end

    end
end



