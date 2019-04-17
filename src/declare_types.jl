using LinearAlgebra
using LinearAlgebra.BLAS: BlasFloat,BlasReal   # Union{Complex{Float32}, Complex{Float64}, Float32, Float64}
using LinearAlgebra:checksquare,AbstractTriangular

BlasVector{T} = AbstractArray{T,1} where  T <: BlasFloat   
BlasMatrix{T} = AbstractArray{T,2} where  T <: BlasFloat   
BlasArray{T} = Union{BlasVector{T},BlasMatrix{T}}

BlasTranspose{T} = Transpose{T,P} where P <: BlasArray{T} 
BlasAdjoint{T} = Adjoint{T,P} where P <: BlasArray{T} 
BlasRow{T} = Transpose{T,P} where P <: BlasVector{T}

BlasNode{T,N} = Union{Array{T,N},SubArray{T,N}} where T <:BlasFloat #add more here if needed
blasnode(M::BlasNode) = M  # it's here
blasnode(M::BlasArray) = blasnode(parent(M))  # go looking for it deeper down ...
