export add_update!

tr2blas(X::BlasArray) = 'N', X
tr2blas(X::BlasTranspose) = 'T', parent(X)
tr2blas(X::BlasAdjoint) = 'C', parent(X)
tr2blas(X::BlasAdjoint{T}) where T <: BlasReal = 'T', parent(X)


#  GEMM and SYMM =======================


do_gemm!(α, TA, A, TB, B, β, C) = BLAS.gemm!(TA, TB, α, A, B, β, C)      # C ← αAB + βC  (with transpositions as specified)
do_gemm!(α, TA, A::Symmetric, TB, B, β, C) = BLAS.symm!('L', A.uplo, α, blasnode(A), B, β, C)
do_gemm!(α, TA, A, TB, B::Symmetric, β, C) = BLAS.symm!('R', B.uplo, α, blasnode(B), A, β, C)


function gemm_αABβC!(α, A, B, β, C::BlasMatrix{T}) where T
    (C===blasnode(A) || C===blasnode(B)) && ip_error("multiplicative and additive updates cannot be combined.")
    try 
        α, β = convert.(T, (α, β))
        do_gemm!(α, tr2blas(A)..., tr2blas(B)..., β, C) 
    catch err
       ip_error(err)
   end
end


add_update0!(C::BlasMatrix{T}, β::Number,  α::Number, A::BlasMatrix{T}, B::BlasMatrix{T}) where T = 
    gemm_αABβC!(α, A, B, β, C)
#




# AXPY and AXPBY and copyto!  ===================================================================== 
for (pm,sα) in ( (:+, :α), (:-, :(-α)) )
    @eval begin
        function add_update!(y::AxpyVec{T}, β::Number, ::typeof($pm), α::Number, x::AxpyVec{T}) where T 
            try 
                β==0 && α==1 && return copyto!(y,x)
                α, β = convert.(T, ($sα, β))
                β==1 ? BLAS.axpy!(α,x,y) : BLAS.axpby!(α,x,β,y)  
            catch err
                ip_error(err)
            end 
        end
    end
end
add_update!(y::AxpyVec{T}, β::Number, pm::Function, x::AxpyVec{T}, α::Number) where T = 
    add_update!(y::AxpyVec{T}, β::Number, pm::Function, α::Number, x::AxpyVec{T})
#
add_update!(y::AxpyVec{T}, β::Number, pm::Function, x::AxpyVec{T}) where T = 
    add_update!(y::AxpyVec{T}, β::Number, pm::Function, 1, x::AxpyVec{T})
#


# GEMV and SYMV =====================================================================================
import LinearAlgebra: lmul!,rmul!

do_gemv!(α, TA, A, B, β, C) = BLAS.gemv!(TA, α, A, B, β, C)      # C ← αAB + βC  (with transposition as specified)
do_gemv!(α, TA, A::Symmetric, B, β, C) = BLAS.symv!(A.uplo, α, blasnode(A), B, β, C)


function gemv_αABβC!(α, A, B, β, C::BlasVector{T}) where T
    (C===blasnode(A) || C===blasnode(B)) && ip_error("multiplicative and additive updates cannot be combined.")
    try 
        α, β = convert.(T, (α, β))
        do_gemv!(α, tr2blas(A)..., B, β, C) 
    catch err
        ip_error(err)
    end
end

add_update0!(C::BlasVector{T}, β::Number, α::Number, A::BlasMatrix{T}, B::BlasVector{T}) where T = 
    gemv_αABβC!(α, A, B, β, C)
#


# SYRK =============================================
function do_syrk!(α, A, B, β, C::Symmetric{T}) where T 
    TA, A = tr2blas(A) 
    TB, B = tr2blas(B)
    cond1 = A===B
    cond2 = (TA=='N' && TB=='T') || (TA=='T' && TB=='N')  
    ( cond1 && cond2) || ip_error("conditions violated for update to Symmetric LHS.")
    try 
        α, β = convert.(T, (α, β))
        BLAS.syrk!(C.uplo, TA, α, A, β, blasnode(C))
        return C  # syrk! returns parent(C), which is usually not symmetric
    catch err
        ip_error(err)
    end
end

add_update0!(C::Symmetric{T}, β::Number, α::Number, A::BlasMatrix{T}, B::BlasMatrix{T}) where T =
    do_syrk!(α, A, B, β, C)
#

# HERK =============================================
function do_herk!(α, A, B, β, C::Hermitian{T}) where T  <: Complex{F} where F
    TA, A = tr2blas(A) 
    TB, B = tr2blas(B)
    cond1 = A===B
    cond2 = (TA=='N' && TB=='C') || (TA=='C' && TB=='N')  
    ( cond1 && cond2) || ip_error("conditions violated for update to Hermitian LHS.")
    try 
        α, β = convert.(F, (α, β))
        BLAS.herk!(C.uplo, TA, α, A, β, blasnode(C))
        return C  # herk! returns parent(C)
    catch err
        ip_error(err)
    end
end

add_update0!(C::Hermitian{T}, β::Number, α::Number, A::BlasMatrix{T}, B::BlasMatrix{T}) where T <: Complex = 
    do_herk!(α, A, B, β, C)
#





# SYR =============================================
function do_syr!(α, A, B, β, C::Symmetric{T}) where T
    β==1 || ip_error("symmetric rank 1 update cannot prescale LHS (β==1 required).")
    TB, B = tr2blas(B)
    (A===B && TB=='T') || ip_error("conditions violated for update to Symmetric LHS.")
    try 
        α = convert(T, α)
        BLAS.syr!(C.uplo, α, A, blasnode(C))
        return C  # syr! returns parent(C), which is usually not symmetric
    catch err
        ip_error(err)
    end
end

add_update0!(C::Symmetric{T}, β::Number, α::Number, A::BlasVector{T}, B::BlasRow{T}) where T = 
    do_syr!(α, A, B, β, C)
#


# HER =============================================
function do_her!(α, A, B, β, C::Hermitian{T}) where T  <: Complex{F} where F
    β==1 || ip_error("Hermitian rank 1 update cannot prescale LHS (β==1 required).")
    A===parent(B) || ip_error("conditions violated for update to Hermitian LHS.")
    try 
        α = convert(F, α)
        BLAS.her!(C.uplo, α, A, blasnode(C))
        return C  # her! returns parent(C)
    catch err
        ip_error(err)
    end
end

add_update0!(C::Hermitian{T}, β::Number, α::Number, A::BlasVector{T}, B::BlasAdjRow{T}) where T <: Complex = 
    do_her!(α, A, B, β, C)
#    



add_update!(C, β, pm::Function, α, A, B) = ip_error("inplace assignment not available for this combination of types.")
add_update0!(C, β::Number, α::Number, A, B) = ip_error("inplace assignment not available for this combination of types.")
for (pm, sα, s) in ( (:+, :α, 1), (:-, :(-α), -1) )
    @eval begin
        add_update!(C, β::Number, ::typeof($pm), α::Number, A, B) = 
            add_update0!(C, β, $sα, A, B) 
        #
        add_update!(C, β::Number, ::typeof($pm), A, α::Number, B) = 
            add_update0!(C, β, $sα, A, B) 
        #
        add_update!(C, β::Number, ::typeof($pm), A, B, α::Number) = 
            add_update0!(C, β, $sα, A, B) 
        #

        add_update!(C::BlasMatrix{T}, β::Number, ::typeof($pm), A::BlasArray{T}, B::BlasArray{T}) where T = 
            add_update0!(C, β, $s, A, B)
        #


    end
end



