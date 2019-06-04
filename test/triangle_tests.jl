

funs = (lmul!, rmul!, ldiv!, rdiv!)
altfuns = ( (D,F)->F*D, (D,F)->D*F, (D,F)->F\D, (D,F)->D/F)
#triangles = subtypes(LinearAlgebra.AbstractTriangular)
triangles = (LowerTriangular,UpperTriangular,UnitLowerTriangular,UnitUpperTriangular)
flips = (identity,Transpose,Adjoint)

let 
    count = 0
    n = 4
    for (fun,altfun) in zip(funs,altfuns)
        for (Dtype,Ftype) in Iterators.product(triangles,triangles)
            for (Dflip,Fflip) in Iterators.product(flips,flips)
                if Dflip != identity || rand() > 1/2    # test real and complex, but complex only for non-transposed desitinations 
                    D = Dflip(Dtype(randn(n,n)))
                    F = Fflip(Ftype(randn(n,n)))
                else
                    D = Dflip(Dtype(complex.(randn(n,n),randn(n,n))))
                    F = Fflip(Ftype(complex.(randn(n,n),randn(n,n))))
                end

                println("Testing: triangle_update!(",typeof(D),", ",typeof(F),", ",fun," )")

                D0 = copyto!(similar(D),D)
                F0 = copyto!(similar(F),F)
                α = rand() > 1/2 ? 1 : π
                LHS = α == 1 ? triangle_update!(D,F,fun) : triangle_update!(D,F,fun,α)
                @test F == F0  # check F not clobbered
                @test LHS ≈ α*altfun(D0,F)
                count += 1
            end
        end
    end
    println("Perfomed ",count," tests on triangle_update!")        
end
