# InplaceLinalg.jl

[![Build Status](https://travis-ci.org/davidavdav/InplaceLinalg.jl.svg?branch=master)](https://travis-ci.org/davidavdav/InplaceLinalg.jl)

This is a small macro package for Julia to make it somewhat easier to write in-place matrix arithmetic. 

## Install

```julia
pkg> add https://github.com/davidavdav/InplaceLinalg.jl
```

## Problem description

Julia has the `.=` broadcast operator, that appears to do the assignments in-place in certain situations.  For instance, if `C` is a `NxM` matrix and `x` is a vector of length `N`, the broadcasting assignment
```julia
C .= x
```
has very little overhead in memory allocation, and therefore is quite efficient.  In the case `B` is also an `NxM` matrix, the broadcasting
```julia
C .= B
```
is also memory efficient.  However, if the RHS is a matrix multiplication, the intermediate result is first stored in newly allocated memory, and is then copied in-place to `C`, as in 
```julia
C .= A * B
```
In principle, one can write 
```julia
LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, C)
```
to compute `A * B` and store the result in-place in `C`, but that looks somewhat clumsy.  

This macro package let you write `gemm!()` calls in a hopefully more visually pleasing way. 

## Supported in-place matrix multiplication syntax

If `C` is a result matrix of appropriate type and size, and `A` and `B` are matrices, you can write
```julia
using InplaceLinalg

@inplace C = A * B
@inplace C += A * B
@inplace C = A' * B
@inplace C += A' * B
@inplace C = A * B'
@inplace C += A * B'
@inplace C = A' * B'
@inplace C += A' * B'

## The following examples should also work for `+=` and transpositions of A and B
@inplace C = 2 * A * B
@inplace C = 2.0 * A * B 
@inplace C = 2.0f0 * A * B
@inplace C = 2.0 * π * A * B
@inplace C = 2π * A * B
```
and these will be translated to calls to `gemm!(At, BT, α, A, B, β, C)` with the correct options for the transposition characters `AT`, `BT`, arguments `A`, `B`, `C` and factors `α`, `β`. 

Because the way the macro works, parenthesized expressions can be used for `A` and `B` as well, but of course these expressions will themselves allocate memory, defeating the purpose of this macro a little.

## Array scaling

You can scale a dense matrix (of any dimension) using the syntax 
```
@inplace C *= factor
```
which will be translated to a call to `scal!(length(C), factor, C, 1)`. 

## Limitations

 - Currenlty the macro expansion assumes expressions `A` and `B` are both matrices (of compatible size), and therefore always returns a call to `gemm!()`.  However, `gemm!()` appears to be generic enough to be able to deal with either `A` or `B` to be vectors if `C` is an `Array` of the appropriate dimension. 

- Expressions like `C = 2A * B` are memory-inefficient, because of the way this expression is parsed by julia as `C = (2*A) * B`

- LHS indexing like `C[2:2:end] *= -1` is not supported yet

- Other BLAS functions like `ger!()` etc. are not yet supported.  
