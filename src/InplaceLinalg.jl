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
    lhs, ass, rhs = assignment(expr)
    term1, term2 = terms(rhs)
    if term1 != 0
        β, C = factors(term1, 2)
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

function factors(expr::Expr, n=3)
    ret = []
    scalars = []
    if expr.head == :call && expr.args[1] == :*
        for i in 2:min(length(expr.args), n)
            factor = pop!(expr.args)
            scalar, factor = factors(factor, 2)
            pushfirst!(ret, factor)
            scalar != 1 && pushfirst!(scalars, scalar)
        end
        while length(expr.args) > 1
            scalar, = factors(pop!(expr.args), 1)
            scalar != 1 && pushfirst!(scalars, scalar)
        end
        scalar = length(scalars) == 0 ? 1 : length(scalars) == 1 ? scalars[1] : Expr(:call, :*, scalars...)
        return tuple(scalar, ret...)
    else
        return fill(1, n-1)..., expr
    end
end
factors(x, n=3) = fill(1, n-1)..., x


        
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
            α, num = factors(num, 2)
            return α, num, /, den
        else
            if expr.args[1] == :\
                den, num = expr.args[2:3]
                α, num = factors(num, 2)
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
