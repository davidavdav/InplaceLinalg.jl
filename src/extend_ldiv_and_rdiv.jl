import LinearAlgebra: ldiv!, rdiv!

ldiv!(α::Number, B::BlasArray) = lmul!(1 / α, B)
rdiv!(B::BlasArray, α::Number) = rmul!(B, 1 /α )

ldiv!(U::UniformScaling, B::BlasArray) = lmul!(1 / U.λ, B)
rdiv!(B::BlasArray, U::UniformScaling) = rmul!(B, 1 / U.λ)


# unwrap to-be-updated triangular numerators by sending ldiv! to !rdiv and vice-versa
# for (LoTr,UpTr) in ( (LowerTriangular, UpperTriangular), 
#                      (UpperTriangular, LowerTriangular),
#                      (UnitLowerTriangular, UnitUpperTriangular), 
#                      (UnitUpperTriangular, UnitLowerTriangular) 
#                    )
#     for (tra,Tra) in ( (transpose,Transpose), (adjoint,Adjoint) )
#         @eval begin
#             ldiv!(L::$LoTr{T}, R::$Tra{T,<:$UpTr{T}}) where T = 
#                 $tra(rdiv!($tra(R),$tra(L)))
#             #
#             rdiv!(L::$Tra{T,<:$LoTr{T}}, R::$UpTr{T}) where T = 
#                 $tra(ldiv!($tra(R),$tra(L)))
#             #
#         end
#     end
# end
