using LinearAlgebra
using LinearAlgebra.BLAS: BlasFloat, BlasReal   # Union{Complex{Float32}, Complex{Float64}, Float32, Float64}
using LinearAlgebra: checksquare, AbstractTriangular

BlasVector{T} = AbstractArray{T,1} where  T <: BlasFloat   
BlasMatrix{T} = AbstractArray{T,2} where  T <: BlasFloat   
BlasArray{T} = Union{BlasVector{T}, BlasMatrix{T}}

BlasTranspose{T} = Transpose{T,P} where P <: BlasArray{T} 
BlasAdjoint{T} = Adjoint{T,P} where P <: BlasArray{T} 
BlasTransRow{T} = Transpose{T,P} where P <: BlasVector{T}
BlasAdjRow{T} = Adjoint{T,P} where P <: BlasVector{T}
BlasRow{T} = Union{BlasTransRow{T}, BlasAdjRow{T}}

BlasNode{T,N} = Union{Array{T,N}, SubArray{T,N}} where T <:BlasFloat #add more here if needed

TransposeTriangular{T} = Transpose{T,P} where P <: AbstractTriangular{T} where T <: BlasFloat
AdjointTriangular{T} = Adjoint{T,P} where P <: AbstractTriangular{T} where T <: BlasFloat
SimpleTriangular{T} = Union{AbstractTriangular{T}, TransposeTriangular{T}, AdjointTriangular{T}} where T <: BlasFloat
#InverseTriangular{T} = Inversion{P} where P <: SimpleTriangular{T} where T <: BlasFloat
#TransformedTriangular{T} = Union{TransposeTriangular{T}, AdjointTriangular{T},InverseTriangular{T}}
TransformedTriangular{T} = Union{TransposeTriangular{T}, AdjointTriangular{T}}

BlasTriangular{T} = Union{AbstractTriangular{T}, TransformedTriangular{T}, UniformScaling{T}} where T <: BlasFloat 


AxpyVec{T} = Union{DenseArray{T},AbstractVector{T}} where T <: BlasFloat

transposechar(::AbstractMatrix) = 'N'
transposechar(::Transpose) = 'T'
transposechar(::Adjoint{T}) where T <: Real = 'T'
transposechar(::Adjoint{T}) where T <: Complex = 'C'
#transposechar(T::InverseTriangular) = transposechar(parent(T))

blasnode(M::BlasNode) = M  # it's here
blasnode(M::TransformedTriangular) = blasnode(parent(M))  
blasnode(M::BlasArray) = blasnode(parent(M))  # go looking for it deeper down ...

uplochar(T::UnitLowerTriangular) = 'L'
uplochar(T::LowerTriangular) = 'L'
uplochar(T::UnitUpperTriangular) = 'U'
uplochar(T::UpperTriangular) = 'U'
uplochar(T::TransformedTriangular) = uplochar(parent(T))

diagchar(T::UnitLowerTriangular) = 'U'
diagchar(T::LowerTriangular) = 'N'
diagchar(T::UnitUpperTriangular) = 'U'
diagchar(T::UpperTriangular) = 'N'
diagchar(T::TransformedTriangular) = diagchar(parent(T))

getparent(T::AbstractTriangular{F}) where F <: BlasFloat = parent(T)
getparent(T::BlasTriangular) = getfield(T,:parent)
getparent(T::UniformScaling) = getfield(T,:λ)


Base.propertynames(T::BlasTriangular) = (:uplo, :diag, :trans, :parent, :blasnode, fieldnames(typeof(T))...)
Base.getproperty(T::BlasTriangular, p::Symbol) = p==:uplo     ? uplochar(T)       :
                                                 p==:diag     ? diagchar(T)       :
                                                 p==:parent   ? getparent(T)      :
                                                 p==:trans    ? transposechar(T)  :
                                                 p==:blasnode ? blasnode(T)       :
                                                 getfield(T,p)
#


#adj2tr(X) = X
#adj2tr(X::Adjoint{T}) where T <: Real = transpose(parent(X))