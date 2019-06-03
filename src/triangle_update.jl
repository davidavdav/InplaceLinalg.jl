export triangle_update!

function prepare_dest!(D::LowerTriangular{T}) where T 
    n = size(D,1)
    P = parent(D)
    for j=2:n 
        for i=1:j-1
            @inbounds P[i,j] = zero(T)
        end
    end
    return P
end

function prepare_dest!(D::UpperTriangular{T}) where T 
    n = size(D,1)
    P = parent(D)
    for j=1:n-1 
        for i=j+1:n
            @inbounds P[i,j] = zero(T)
        end
    end
    return P
end

function prepare_dest!(D::UnitLowerTriangular{T}) where T 
    n = size(D,1)
    P = parent(D)
    @inbounds P[1,1] = one(T)
    for j=2:n 
        for i=1:j-1
            @inbounds P[i,j] = zero(T)
        end
        @inbounds P[j,j] = one(T)
    end
    return P
end



function prepare_dest!(D::UnitUpperTriangular{T}) where T 
    n = size(D,1)
    P = parent(D)
    for j=1:n-1 
        @inbounds P[j,j] = one(T)
        for i=j+1:n
            @inbounds P[i,j] = zero(T)
        end
    end
    @inbounds P[n,n] = one(T)
    return P
end

prepare_dest!(D) = D


function wrap_dest(D::UnitLowerTriangular, F::BlasTriangular)
    uplo_eff(F) == 'L' || return parent(D)  # result is full square
    F.diag == 'U' || return LowerTriangular(parent(D))  # result diagonal not unit  
    return D # result still has unit diagonal
end
function wrap_dest(D::UnitUpperTriangular, F::BlasTriangular)
    uplo_eff(F) == 'U' || return parent(D)  # result is full square
    F.diag == 'U' || return UpperTriangular(parent(D))  # result diagonal not unit  
    return D # result still has unit diagonal
end
function wrap_dest(D::AbstractTriangular, F::BlasTriangular)
    uplo_eff(F) == uplo_eff(D) || return parent(D) # result is full square
    return D # result is still triangular
end

import LinearAlgebra.BLAS: trmm!, trsm!
for (fun,blasfun,side) in ( (:lmul!, :trmm!, 'L'), 
                            (:rmul!, :trmm!, 'R'),
                            (:ldiv!, :trsm!, 'L'),
                            (:rdiv!, :trsm!, 'R')
                         )
    @eval begin
        function triangle_update!(Dest::AbstractTriangular{T}, F::BlasTriangular{T}, 
                                  ::typeof($fun), α::Number=1.0) where T
            Dest.blasnode == F.blasnode && error("arguments have common ancestor")
            P = prepare_dest!(Dest)
            α = convert(T,α)
            $blasfun($side, F.uplo, F.trans, F.diag, α, F.blasnode, P)  
            return wrap_dest(Dest,F)
        end
    end
end


for (tra,Tra) in ( (:transpose, :Transpose), (:adjoint, :Adjoint))
    for (fwd,rev) in ( (:lmul!,:rmul!), 
                       (:rmul!,:lmul!), 
                       (:ldiv!,:rdiv!), 
                       (:rdiv!,:ldiv!) 
                     )
        @eval begin 
            function triangle_update!(D::$Tra{T,<:BlasMatrix{T}},
                                    F::BlasTriangular{T}, ::typeof($fwd),
                                    α::Number=1.0) where T
                $tra( triangle_update!($tra(D), $tra(F), $rev, α) )                                  
            end
        end
    end
end
