import Base:showerror
export ip_error
abstract type InplaceException <: Exception end

struct InplaceException1 <: InplaceException
    msg::String
end
showerror(io::IO, e::InplaceException1) = print(io, "InplaceException: ", e.msg)


struct InplaceException2 <: InplaceException
    blas_error::Exception
end

function showerror(io::IO, e::InplaceException2) 
    println(io, "InplaceException: inplace assignment for these arguments has failed.")
    println(io, "Try some other combination, or try plain Julia.")
    println(io, "The original exception was:")
    showerror(io::IO, e.blas_error)
    stacktrace(catch_backtrace())
end


ip_error(msg::String) = throw(InplaceException1(msg))
ip_error(err::Exception) = throw(InplaceException2(err))
