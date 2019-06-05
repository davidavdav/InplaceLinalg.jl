#========================================================================= 
This is a reimplementation of LinearAlgebra: rmul!, lmul!, ldiv!, rdiv!, where both arguments are triangular. 
Note:
- Our triangular matrices can be any of the 4 subtypes of LinearAlgbra.AbstractTriangular, i.e. any of:
        {LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular}
- Adjoints and Transposes of triangular matrices are also treated as triangular.

This code addresses the issues raised here:
https://discourse.julialang.org/t/issues-with-inplace-updates-to-triangular-matrices-using-lmul-rmul-ldiv-and-rdiv/24887

This code uses nested loops to declare (using @eval) a multitude of very specific methods, thus shadowing 
existing implementations in triangular.jl and bidiag.jl, which are problematic in V1.1.1.

Some properties of the new methods are:

- They don't call down into BLAS. All methods use three nested loops to iterate over martrix elements, using 
  getindex! and setindex!.

- They are probably faster than pre-existing methods, for small matrices.

- For large matrices, the new methods can be slower or faster than pre-existing implementations, depending on whether 
  the type combinations would have landed up (quite unpredictably) in triangular.jl or bidiag.jl. 

- The new methods aim to be exhaustive for all combinations of arguments that would allow inplace triangle 
  updates, i.e. where:
  -- The uplo of both arguments agree, and
  -- the element types agree. 
  Note:
  -- The uplo of e.g. Transpose(LowerTriangular) and UnitUpperTriangular would agree.
  -- Transposes and Adjoints of triangular matrices can also serve as to-be-updated destinations.

- No clobbering of triangle parents outside of the intended triangle.

- For L,R triangular and fun ∈ {rmul!, lmul!, ldiv!, rdiv!}: 
      applicable(fun,L,R) && fun(L,R)
  should not crash---except when sizes disagree, or when divisors are singular.   

==========================================================================# 

import LinearAlgebra: rmul!, lmul!, ldiv!, rdiv!




UpperA{T} = Adjoint{T,<:LowerTriangular{T}}
UpperT{T} = Transpose{T,<:LowerTriangular{T}}
upperset = (UpperTriangular,UpperA,UpperT)

UnitUpperA{T} = Adjoint{T,<:UnitLowerTriangular{T}}
UnitUpperT{T} = Transpose{T,<:UnitLowerTriangular{T}}
unit_upperset = (UnitUpperTriangular,UnitUpperT,UnitUpperA)

all_upperset = (upperset..., unit_upperset...)

for (DType,FType) in Iterators.product(all_upperset,all_upperset)
    if DType ∈ upperset # all cases where result is non-unit uppertriangular
        
        @eval begin

            # upper
            function rmul!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = n:-1:1
                    for i = j:-1:1
                        s = D[i,i]*F[i,j]
                        for k = i+1:j
                            s += D[i,k]*F[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # upper
            function lmul!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = 1:n
                    for i = 1:j
                        s = F[i,i]*D[i,j]
                        for k = i+1:j
                            s += F[i,k]*D[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # upper
            function ldiv!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = n:-1:1 
                    (den = F[j,j]) == 0 && error("F is singular")
                    D[j,j] /= den
                    for i = j-1:-1:1
                        s = F[i,j] * D[j,j]
                        for k = i+1:j-1
                            s += F[i,k] * D[k,j]
                        end
                        (den = F[i,i]) == 0 && error("F is singular")
                        D[i,j] = (D[i,j] - s) / den 
                    end
                end
                return D
            end

            # upper
            function rdiv!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for i = 1:n
                    (den = F[i,i]) == 0 && error("F is singular")
                    D[i,i] /= den
                    for j = i+1:n
                        s = D[i,i] * F[i,j]
                        for k = i+1:j-1
                            s += D[i,k] * F[k,j]
                        end
                        (den = F[j,j]) == 0 && error("F is singular")
                        D[i,j] = (D[i,j] - s) / den 
                    end
                end
                return D
            end



        end # eval

    elseif DType ∈ unit_upperset && FType ∈ unit_upperset # all cases where result is unit uppertriangular
        
        @eval begin

            # unit upper
            function rmul!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = n:-1:1
                    for i = j-1:-1:1
                        s = F[i,j]
                        for k = i+1:j
                            s += D[i,k]*F[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # unit upper
            function lmul!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = 1:n
                    for i = 1:j-1
                        s = D[i,j]
                        for k = i+1:j
                            s += F[i,k]*D[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # unit upper
            function ldiv!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = n:-1:1 
                    for i = j-1:-1:1
                        s = F[i,j] 
                        for k = i+1:j-1
                            s += F[i,k] * D[k,j]
                        end
                        D[i,j] -= s
                    end
                end
                return D
            end

            # unit upper
            function rdiv!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for i = 1:n
                    for j = i+1:n
                        s = F[i,j]
                        for k = i+1:j-1
                            s += D[i,k] * F[k,j]
                        end
                        D[i,j] -= s
                    end
                end
                return D
            end


        end  #eval
    
    end # if
end # for


# lower triangular 

LowerA{T} = Adjoint{T,<:UpperTriangular{T}} 
LowerT{T} = Transpose{T,<:UpperTriangular{T}}
lowerset = (LowerTriangular,LowerA,LowerT)

UnitLowerA{T} = Adjoint{T,<:UnitUpperTriangular{T}}
UnitLowerT{T} = Transpose{T,<:UnitUpperTriangular{T}}
unit_lowerset = (UnitLowerTriangular,UnitLowerT,UnitLowerA)

all_lowerset = (lowerset..., unit_lowerset...)

for (DType,FType) in Iterators.product(all_lowerset,all_lowerset)
    if DType ∈ lowerset # all cases where result is non-unit lowertriangular
        
        @eval begin

            # lower
            function rmul!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j=1:n
                    for i=j:n
                        s = D[i,j]*F[j,j]
                        for k=j+1:i
                            s += D[i,k]*F[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # lower
            function lmul!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j=n:-1:1
                    for i=n:-1:j
                        s = F[i,j]*D[j,j]
                        for k=j+1:i
                            s += F[i,k]*D[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # lower
            function ldiv!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = 1:n 
                    (den = F[j,j]) == 0 && error("F is singular")
                    D[j,j] /= den
                    for i = j+1:n
                        s = F[i,j] * D[j,j]
                        for k = j+1:i-1 
                            s += F[i,k] * D[k,j]
                        end
                        (den = F[i,i]) == 0 && error("F is singular")
                        D[i,j] = (D[i,j] - s) / den 
                    end
                end
                return D
            end

            # lower
            function rdiv!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for i = n:-1:1 
                    (den = F[i,i]) == 0 && error("F is singular")
                    D[i,i] /= den
                    for j = i-1:-1:1
                        s = D[i,i] * F[i,j]
                        for k = j+1:i-1
                            s += D[i,k] * F[k,j]
                        end
                        (den = F[j,j]) == 0 && error("F is singular")
                        D[i,j] = (D[i,j] - s) / den 
                    end
                end
                return D
            end

        end #eval 
    
    elseif DType ∈ unit_lowerset && FType ∈ unit_lowerset # all cases where result is unit lowertriangular
    
        @eval begin

            # unit lower
            function rmul!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j=1:n
                    for i=j+1:n
                        s = D[i,j]
                        for k=j+1:i
                            s += D[i,k]*F[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # unit lower
            function lmul!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j=n:-1:1
                    for i=n:-1:j+1
                        s = F[i,j]
                        for k=j+1:i
                            s += F[i,k]*D[k,j]
                        end
                        D[i,j] = s
                    end
                end
                return D
            end

            # unit lower
            function ldiv!(F::$FType{T}, D::$DType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for j = 1:n 
                    for i = j+1:n
                        s = F[i,j] 
                        for k = j+1:i-1 
                            s += F[i,k] * D[k,j]
                        end
                        D[i,j] -= s
                    end
                end
                return D
            end

            # unit lower
            function rdiv!(D::$DType{T}, F::$FType{T}) where {T}
                m,n = size(D,1), size(F,1)
                n == m || error("$m≠$n")
                @inbounds for i = n:-1:1 
                    for j = i-1:-1:1
                        s = F[i,j]
                        for k = j+1:i-1
                            s += D[i,k] * F[k,j]
                        end
                        D[i,j] -= s
                    end
                end
                return D
            end

        end  # eval

    end # if
end # for


