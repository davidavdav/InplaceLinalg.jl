module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(e::Expr)
    ## dump(e)
    if e.head in [:(=), :(+=)] && isa(e.args[1], Symbol) 
        lhs, rhs = e.args
        if isa(rhs, Expr) && rhs.head == :call && length(rhs.args) ≥ 3 && rhs.args[1] == :* 
            tr2, arg2 = InplaceLinalg.trans(pop!(rhs.args))
            tr1, arg1 = InplaceLinalg.trans(pop!(rhs.args))
            call = popfirst!(rhs.args) ##  * 
            alpha = InplaceLinalg.factor(rhs.args, arg1)
            beta = e.head == :(+=) ? :(one(eltype($arg1))) : :(zero(eltype($arg1)))
            return :(BLAS.gemm!($tr1, $tr2, $alpha, $arg1, $arg2, $beta, $lhs))
        elseif isa(rhs, Symbol) && e.head == :(=)
            return :(BLAS.blascopy!(length($rhs), $rhs, 1, $lhs, 1))
        elseif e.head == :(+=)
            if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :* && isa(rhs.args[end], Symbol)
                ## this is supposed to catch C += 2 * A, but that is already dispatched above by gemm!()
                call = popfirst!(rhs.args)
                X = pop!(rhs.args)
                alpha = InplaceLinalg.factor(rhs.args, X)
                return :(BLAS.axpy!($alpha, $X, $lhs))
            else 
                return :(BLAS.axpy!(one(eltype($lhs)), $rhs, $lhs))
            end
        end
    elseif e.head == :(*=) && isa(e.args[1], Symbol)
        lhs = e.args[1]
        factor = :(convert(eltype($lhs), $(e.args[2])))
        return :(BLAS.scal!(length($lhs), $factor, $lhs, 1))
    end
    return e
end

trans(expr::Expr) = (expr.head == Symbol("'")) ? ('T', expr.args[1]) : ('N', expr)
trans(x) = 'N', x

## Evaluate the first factors in a product
function factor(factors::Array, array::Symbol) 
    nfactors = length(factors)
    nfactors == 0 && return :(one(eltype($array)))
    nfactors == 1 && return :(convert(eltype($array), $(factors[1])))
    return :(convert(eltype($array), prod($(factors))))
end

end # module
