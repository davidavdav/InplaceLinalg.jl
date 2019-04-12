module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(e::Expr)
    ## dump(e)
    if e.head == :(=) && isa(e.args[1], Symbol) && isa(e.args[2], Expr) && e.args[2].head == :call && e.args[2].args[1] == :*
        return :(BLAS.gemm!('N', 'N', 1.0, $(e.args[2].args[2]), $(e.args[2].args[3]), 0.0, $(e.args[1])))
    end
    return e
end

end # module
