import LinearAlgebra: lmul!, rmul!

function rmul!(A::AbstractVector, D::Diagonal)
    @assert !LinearAlgebra.has_offset_axes(A)
    A .= A .* transpose(D.diag)
    return A
end

function lmul!(D::Diagonal, B::AbstractVector)
    @assert !LinearAlgebra.has_offset_axes(B)
    B .= D.diag .* B
    return B
end
