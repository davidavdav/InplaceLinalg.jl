module InplaceLinalg

using LinearAlgebra

export @inplace,InplaceException

include("declare_types.jl")

macro inplace(x)
    esc(inplace(x))
end

function inplace(expr::Expr) 
    #dump(expr)
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
    elseif (β, α, typeof(A)) == (0, 1, Expr) && A.args[1] == :\
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



include("error_handling.jl")

C_AB!(C, β, α, A, B) = ip_error(": inplace assignment for this combination of types not implemented.")
include("C_AB.jl")


include("C_div.jl")



end # module
