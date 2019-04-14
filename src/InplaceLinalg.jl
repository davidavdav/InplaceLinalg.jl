module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(e::Expr)
    ## dump(e)
    if e.head in [:(=), :(+=)] && isa(e.args[1], Symbol) 
        rhs = e.args[2]
        if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :*
            tr2, arg2 = trans(pop!(rhs.args))
            tr1, arg1 = trans(pop!(rhs.args))
            call = popfirst!(rhs.args) ##  * 
            nfactors = length(rhs.args)
            alpha = nfactors == 0 ? :(one(eltype($arg1))) : 
                nfactors == 1 ? :(convert(eltype($arg1), $(rhs.args[1]))) : :(convert(eltype($arg1), prod($(rhs.args))))
            beta = e.head == :(+=) ? :(one(eltype($arg1))) : :(zero(eltype($arg1)))
            return :(BLAS.gemm!($tr1, $tr2, $alpha, $arg1, $arg2, $beta, $(e.args[1])))
        end
    end
    return e
end

trans(expr::Expr) = (expr.head == Symbol("'")) ? ('T', expr.args[1]) : ('N', expr)
trans(x) = 'N', x

end # module
