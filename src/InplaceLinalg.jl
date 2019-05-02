module InplaceLinalg

using LinearAlgebra

export @inplace, InplaceException

include("declare_types.jl")
include("error_handling.jl")


macro inplace(x)
    try
        esc(inplace(x))
    catch e
        :(ip_error($(e.msg)))
    end
end

function inplace(expr::Expr) 
    #dump(expr)
    ## parse assignment or err
    lhs, ass, rhs = assignment(expr)
    ## select update expressions involving divisions
    if ass == :(=) && isa(rhs, Expr) && rhs.head == :call && length(rhs.args) == 3
        if rhs.args[1] == :/
            num, den = rhs.args[2:3]
            args = divupdate(lhs, num)
            return :(InplaceLinalg.div_update!($(args...), /, $den))
        elseif rhs.args[1] == :\
            den, num = rhs.args[2:3]
            args = divupdate(lhs, num)
            return :(InplaceLinalg.div_update!($(args...), \, $den))
        end
        ## fall through for other expressions with :=
    elseif ass == :(/=)
        ## accept any RHS
        return :(InplaceLinalg.div_update!($lhs, /, $rhs))
    end
    term1, term2 = terms(rhs)
    if ass == :(=) && term1 == 0 
        n, facs = multupdate(lhs, rhs)
        n > 0 && return :(InplaceLinalg.mult_update!($lhs, $(facs...), Val($n)))
    end
    if term1 != 0
        β, C = factors(term1, 2)
        #@assert lhs == C "First term must be linear in the LHS" lhs C
        lhs == C || return :(ip_error("First term must be a multiple of the LHS"))
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
    ass in [:(/=), :(*=)] && return :(ip_error("Unexpected assignment operator"))
    ## println((β, α, A, B))
    isa(B, Symbol) || 
        isa(B, Expr) && B.head == Symbol("'") && isa(B.args[1], Symbol) || 
        return :(ip_error("Too complex expression for inplace"))
    return :(InplaceLinalg.C_AB!($lhs, $β, $α, $A, $B))
end
inplace(x) = x

function assignment(expr::Expr)
    expr.head in [:(=), :(+=), :(*=), :(/=), :(-=)] || error("Unknown assignment operator in " * string(expr))
    isa(expr.args[1], Symbol) || error("LHS should be a symbolic variable")
    return expr.args[1], expr.head, expr.args[2]
end

function divupdate(lhs::Symbol, num::Symbol)
    lhs == num || error("LHS must be equal to numerator in updating divide")
    return tuple(num)
end
function divupdate(lhs::Symbol, num::Expr)
    num.head == :call && length(num.args) == 3 && num.args[1] == :* || error("Numerator must be simple multiplicative expression")
    matches = lhs .== num.args[2:3]
    sum(matches) == 1 || error("LHS must appear exactly once in multiplicative numerator expression")
    return tuple(num.args[2 .+ .!matches]...) ## return args in correct order
end

function terms(expr::Expr)
    if expr.head == :call && length(expr.args) == 3 && expr.args[1] == :+
        return expr.args[2:3]
    else
        return 0, expr
    end
end
terms(x) = 0, x

function multupdate(lhs::Symbol, rhs::Expr)
    if rhs.head == :call && length(rhs.args) ≤ 4 && rhs.args[1] == :* 
        factors = rhs.args[2:end]
        matches = lhs .== factors
        sum(matches) == 1 && return findfirst(matches), factors[.!matches]
    end
    return 0, rhs
end
multupdate(lhs::Symbol, rhs) = 0, rhs

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



C_AB!(C, β, α, A, B) = ip_error(": inplace assignment for this combination of types not implemented.")
C_div!(C, α, B, div, A) = ip_error(": inplace assignment for this combination of types not implemented.")

include("extend_ldiv_and_rdiv.jl")
include("extend_lmul_and_rmul.jl")

include("C_AB.jl")

include("div_update.jl")
include("mult_update.jl")



end # module
