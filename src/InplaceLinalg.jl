module InplaceLinalg

using LinearAlgebra

export @inplace, InplaceException

include("declare_types.jl")
include("error_handling.jl")




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
        #@assert lhs == C "First term must be linear in the LHS" lhs C
        lhs ==C || return :(ip_error("First term must be a multiple of the LHS"))
    else 
        β = 0
    end
    α, A, B = factors(term2)
    if ass in [:(+=), :(-=)]
        β = dobeta(Val{ass}, β)
        if ass == :(-=)
            α = negate(α)
        end
        ass = :(=)
    end
    if (β, α, A) == (0, 1, 1)
        α, B, div, A = quotient(B)
        if (div != nothing) 
            ass == :(=) && return :(InplaceLinalg.C_div!($lhs, $α, $B, $div, $A))
            return :(ip_error("Can only use / or \\ with plain assignment ="))
        else
            ass == :(/=) && return :(InplaceLinalg.C_div!($lhs, 1, $lhs, $(/), $B))
            ass == :(=) && return :($lhs .= $B)
        end
        return :(ip_error("Unhandled case"))
    elseif (β, α, typeof(A)) == (0, 1, Expr) && A.args[1] == :\
        γ, α, div, A = quotient(A)
        if γ != 1
            α = :($γ * $α)
        end
        return :(InplaceLinalg.C_div!($lhs, $α, $B, $div, $A))
    end
    ass in [:(/=), :(*=)] && return :(ip_error("Unexpected assignment operator"))
    ## println((β, α, A, B))
    isa(B, Symbol) || 
        isa(B, Expr) && B.head == Symbol("'") && isa(B.args[1], Symbol) || 
        return :(ip_error("Too complex expression for inplace"))
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

function factors(expr::Expr, min=3)
    if expr.head == :call && length(expr.args) ≥ min && expr.args[1] == :*
        B = pop!(expr.args)
        A = pop!(expr.args)
        _, a, A = factors(A, 2)
        _, b, B = factors(B, 2)
        for f in [a, b]
            f != 1 && push!(expr.args, f)
        end
        alpha = InplaceLinalg.factor(expr)
        return alpha, A, B
    else
        return 1, 1, expr
    end
end
factors(x, min=3) = 1, 1, x


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
                _, α, num = factors(num, 2)
                return α, num, \, den
            end
        end
    end
    return 1, expr, nothing, nothing
end
quotient(x) = 1, x, nothing, nothing




C_AB!(C, β, α, A, B) = ip_error(": inplace assignment for this combination of types not implemented.")
C_div!(C, α, B, div, A) = ip_error(": inplace assignment for this combination of types not implemented.")

include("C_AB.jl")
include("C_div.jl")



end # module
