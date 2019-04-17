module InplaceLinalg

using LinearAlgebra: BLAS

export @inplace, BLAS

macro inplace(x)
    esc(inplace(x))
end

function inplace(expr::Expr) 
    lhs, ass, rhs = assignment(expr);
    term1, term2 = terms(rhs)
    if term1 != 0
        α, β, C = factors(term1)
        if α != 1
            β = :($α * $β)
        end
        @assert lhs == C "First term must be linear in the LHS" lhs C
    else 
        β = 0
    end
    α, A, B = factors(term2)
    if ass in [:(+=), :(-=)]
        β = dobeta(Val{ass}, β)
    end
    if ass == :(-=)
        α = negate(α)
    end
    if (β, α, A) == (0, 1, 1)
        α, B, div, A = quotient(B)
        if (div != nothing)
            return :(InplaceLinalg.C_div($lhs, $α, $B, $div, $A))
        else
            return :($lhs .= $B)
        end
    elseif (β, α, typeof(A)) == (0, 1, Expr)
        _, α, div, A = quotient(A)
        return :(InplaceLinalg.C_div($lhs, $α, $B, $div, $A))
    end
    return :(InplaceLinalg.C_AB!($lhs, $β, $α, $A, $B))
end
inplace(x) = x

function assignment(expr::Expr) 
    @assert expr.head in [:(=), :(+=), :(*=), :(/=), :(-=)] "Unknown assignment operator " * string(expr)
    return expr.args[1], expr.head, expr.args[2]
end

function terms(expr::Expr)
    if expr.head == :call && length(expr.args) == 3 && expr.args[1] == :+
        return expr.args[2:3]
    else
        return 0, expr
    end
end
terms(x) = 0, x

function factors(expr::Expr)
    if expr.head == :call && length(expr.args) ≥ 3 && expr.args[1] == :*
        B = pop!(expr.args)
        A = pop!(expr.args)
        alpha = InplaceLinalg.factor(expr)
        return alpha, A, B
    else
        return 1, 1, expr
    end
end
factors(x) = 1, 1, x


function factor(expr::Expr)
    nfactors = length(expr.args) - 1
    nfactors == 0 && return 1
    nfactors == 1 && return expr.args[2]
    return expr
end

dobeta(::Type{Val{:(+=)}}, x::Number) = x + 1
dobeta(::Type{Val{:(-=)}}, x::Number) = (1 - x)
dobeta(::Type{Val{:(+=)}}, x) = :($x + 1)
dobeta(::Type{Val{:(-=)}}, x) = :(1 - $x)

negate(x::Number) = -x
negate(x) = :(-$x)

function quotient(expr::Expr)
    if expr.head == :call && length(expr.args) == 3
        if expr.args[1] == :/
            num, den = expr.args[2:3]
            γ, α, num = factors(num)
            if γ != 1
                α = :($γ * $α)
            end
            return α, num, /, den
        else
            if expr.args[1] == :\
                den, num = expr.args[2:3]
                return 1, num, \, den
            end
        end
    end
    return 1, expr, nothing, nothing
end
quotient(x) = 1, x, nothing, nothing

function C_AB!(C, β, α, A, B)
    return C, β, α, typeof(A), A, typeof(B), B
end

function C_div!(C, α, B, div, A)
    return C, typeof(α), α, typeof(B), B, A
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
