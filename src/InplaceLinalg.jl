module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(expr::Expr) 
    if expr.head in [:(=), :(+=)] && isa(expr.args[1], Symbol)
        lhs, rhs = expr.args
        if isa(rhs, Expr) && rhs.head == :call && length(rhs.args) ≥ 3 
            if rhs.args[1] == :* 
                B = pop!(rhs.args)
                A = pop!(rhs.args)
                alpha = InplaceLinalg.factor(rhs)
                beta = Int(expr.head == :(+=))
                return :(InplaceLinalg.C_AB!($lhs, $beta, $alpha, $A, $B))
            elseif rhs.args[1] == :+ && length(rhs.args) == 3
                term1, term2 = rhs.args[2:3]
                beta, C = InplaceLinalg.firstTerm(term1)
                @assert lhs == C "First term must be linear in the LHS"
                if isa(term2, Expr) && term2.head == :call && length(term2.args) ≥ 3 && term2.args[1] == :*
                    B = pop!(term2.args)
                    A = pop!(term2.args)
                    alpha = InplaceLinalg.factor(term2)
                    return :(InplaceLinalg.C_AB!($lhs, $beta, $alpha, $A, $B))
                end
            end
        end 
    end
    error("Unsupported expression")
end

firstTerm(term::Symbol) = 1, term
function firstTerm(term::Expr) 
    @assert term.head == :call && length(term.args) ≥ 3 && term.args[1] == :* "First term must be simple multiplication"
    return term.args[2], term.args[3]
end

function factor(expr::Expr)
    nfactors = length(expr.args) - 1
    nfactors == 0 && return 1
    nfactors == 1 && return expr.args[2]
    return expr
end

function C_AB!(C, β, α, A, B)
    return C, β, α, typeof(A), A, typeof(B), B
end

## Old stuff

function oldinplace(e::Expr)
    ## dump(e)
    if e.head in [:(=), :(+=)] && isa(e.args[1], Symbol) 
        lhs, rhs = e.args
        if isa(rhs, Expr) && rhs.head == :call && length(rhs.args) ≥ 3 && rhs.args[1] == :* 
            tr2, arg2 = InplaceLinalg.trans(pop!(rhs.args))
            tr1, arg1 = InplaceLinalg.trans(pop!(rhs.args))
            alpha = InplaceLinalg.factor(rhs, arg1)
            beta = e.head == :(+=) ? :(one(eltype($arg1))) : :(zero(eltype($arg1)))
            return :(BLAS.gemm!($tr1, $tr2, $alpha, $arg1, $arg2, $beta, $lhs))
        elseif isa(rhs, Symbol) && e.head == :(=)
            return :(BLAS.blascopy!(length($rhs), $rhs, 1, $lhs, 1))
        elseif e.head == :(+=)
            if isa(rhs, Expr) && rhs.head == :call && rhs.args[1] == :* && isa(rhs.args[end], Symbol)
                ## this is supposed to catch C += 2 * A, but that is already dispatched above by gemm!()
                X = pop!(rhs.args)
                alpha = InplaceLinalg.factor(rhs, X)
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
function factor(expr::Expr, array::Symbol) 
    nfactors = length(expr.args) - 1
    nfactors == 0 && return :(one(eltype($array)))
    nfactors == 1 && return :(convert(eltype($array), $(expr.args[2])))
    return :(convert(eltype($array), $expr))
end

end # module
