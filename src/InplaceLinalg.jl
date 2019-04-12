module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(e::Expr)
    ## dump(e)
    if e.head == :(=) && isa(e.args[1], Symbol) && isa(e.args[2], Expr) && e.args[2].head == :call && e.args[2].args[1] == :*
        tr1, arg1 = trans(e.args[2].args[2])
        tr2, arg2 = trans(e.args[2].args[3])
        return :(BLAS.gemm!($tr1, $tr2, 1.0, $arg1, $arg2, 0.0, $(e.args[1])))
    end
    return e
end

trans(expr::Expr) = (expr.head == Symbol("'")) ? ('T', expr.args[1]) : ('N', expr)
trans(x) = 'N', x

end # module
