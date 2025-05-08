# Solving the dynamical formulation of RMMOT


using TimerOutputs
using LinearAlgebra
using SparseArrays
using Plots


# ------------------------------------------------------------
# Build matrices
# ------------------------------------------------------------



function matrixAverageLinear(nSize)
    # Matrix corresponding to average
    output = spzeros(nSize-1,nSize)
    for i = 1:nSize-1
        output[i,i] = 0.5
        output[i,i+1] = 0.5
    end
    return output
end

function matrixAverageLinearPer(nSize)
    # Matrix corresponding to average with periodic boundary conditions
    output = spzeros(nSize,nSize)
    for i = 1:nSize-1
        output[i,i] = 0.5
        output[i,i+1] = 0.5
    end
    output[end,end] = 0.5
    output[end,1] = 0.5
    return output
end


function matrixDerivativeLinear(nSize,delta)
    # Matrix corresponding to derivative 
    output = spzeros( nSize-1, nSize )
    for i = 1:nSize-1
        output[i,i+1] = 1/delta
        output[i,i] = - 1/delta
    end
    return output
end

function matrixDerivativeLinearPer(nSize,delta)
    # Matrix corresponding to derivative with periodic boundary conditions
    output = spzeros( nSize, nSize )
    for i = 1:nSize-1
        output[i,i+1] = 1/delta
        output[i,i] = - 1/delta
    end
    output[end,end] = -1/delta
    output[end,1] = 1/delta
    return output
end


function matrixLaplacianLinearPer(nSize,delta)
    # Matrix corresponding to Laplacian with periodic boundary conditions
    output = spzeros( nSize, nSize )
    for i = 1:nSize
        output[i,i] = - 2
        if i <= nSize - 1
            output[i,i+1] = 1
        else
            output[i,1] = 1
        end
        if i >= 2
            output[i,i-1] = 1
        else
            output[i,end] = 1
        end
    end
    return 1/delta^2 * output
end



function constructFullMatrix(A,B)
    # If A is a matrix which acts on R^n and B a matrix which acts on R^m,
    # returns a matrix which acts on R^{nm} with the right ordering (tensor product of matrices)

    rowA = size(A)[1]
    colA = size(A)[2]
    rowB = size(B)[1]
    colB = size(B)[2]

    # Define the full matrix
    output = spzeros(rowA*rowB,colA*colB)

    # Linear and cartesian indices
    linearR = LinearIndices((1:rowA, 1:rowB))
    linearC = LinearIndices((1:colA, 1:colB))

    # Fill the matrix
    for indexA in findall(!iszero, A),  indexB in findall(!iszero, B)
        output[linearR[indexA[1],indexB[1]],linearC[indexA[2],indexB[2]]] = A[indexA]*B[indexB]
    end

    return output
end


function constructFullVector(A,B)
    # If A in R^n and B in R^m, return the vector A[x] B[y] with the right ordering (tensor product of vectors)

    lengthA = size(A)[1]
    lengthB = size(B)[1]

    # Define the full matrix
    output = zeros(lengthA*lengthB)

    # Linear and cartesian indices
    linearIndex = LinearIndices((1:lengthA, 1:lengthB))

    # Fill the vector
    for i = 1:lengthA, j = 1:lengthB,
        output[linearIndex[i,j]] = A[i]*B[j]
    end

    return output
end


function constructDimensionaFullMatrix(A, dimension, structure)
    # A is n x n matrix. Returns tensor products of A with identity I_n with to powers d = dimension as follows.
    # If "structure" = "concatenated" returns an n^d by d(n^d) matrix given by
    # [A \otimes I_{n^{d-1}}, ..., I_{n^i}\otimes A \otimes I_{n^{d-i}}, ..., I_{n^{d-1}} \otimes A ]
    # If "structure" = "diagonal" returns a d(n^d) by d(n^d) matrix given by
    # [A \otimes I_{n^{d-1}}, 0,...,0;
    # 0, 0, ..., I_{n^i}\otimes A \otimes I_{n^{d-i-1}, 0,...,0;
    # 0, 0, ..., I_{n^{d-1}} \otimes A ]

    if !(structure in ("concatenated", "diagonal"))
        error("The structure of the matrix is not recognized")
    end
    nSize = size(A)[1]
    if structure == "concatenated"
        output = spzeros(nSize^dimension , dimension*nSize^dimension)
    elseif structure == "diagonal"
        output = spzeros(dimension*nSize^dimension , dimension*nSize^dimension) 
    end
    for i = 0 : dimension-1
        if i == 0
            outputPartial = constructFullMatrix( A,  sparse(I,nSize^(dimension-1),nSize^(dimension-1)))
            if structure == "concatenated"
                output[1:nSize^dimension, 1:nSize^dimension] = outputPartial
            elseif structure == "diagonal"
                output[1:nSize^dimension, 1:nSize^dimension] = outputPartial
            end
        elseif i == dimension-1
            outputPartial = constructFullMatrix( sparse(I,nSize^(dimension-1),nSize^(dimension-1)), A )
            if structure == "concatenated"
                output[1:nSize^dimension, 1 + (dimension-1)*nSize^dimension : dimension*nSize^dimension] = outputPartial
            elseif structure == "diagonal"
                output[(dimension-1)*nSize^dimension+1:end, 1 + (dimension-1)*nSize^dimension : end] = outputPartial
            end
        else
            outputPartial = constructFullMatrix(constructFullMatrix(sparse(I,nSize^i, nSize^i), A ), sparse(I,nSize^(dimension-i-1), nSize^(dimension-i-1)))
            if structure == "concatenated"
                output[1:nSize^dimension, 1+i*nSize^dimension : (i+1)*nSize^dimension] = outputPartial
            elseif structure == "diagonal"
                output[1+i*nSize^dimension : (i+1)*nSize^dimension, 1+i*nSize^dimension : (i+1)*nSize^dimension] = outputPartial
            end
        end
    end
    return output
end




function matrixMarginalConstraints(nSize, nMarginals)
    # Marginal constraints for dynamical multi-dimensional optimal transport with nMarginals in dimension nSize
    output = zeros(nMarginals * nSize, nSize^nMarginals )
    for i = 1:nMarginals
        if i == 1
            output[1:nSize, 1: end ] = constructFullMatrix(Matrix(I, nSize, nSize), ones(1, nSize^(nMarginals-1)))
        elseif i == nMarginals
            output[(nMarginals-1)*nSize+1: nMarginals * nSize, 1: end ] = constructFullMatrix(ones(1, nSize^(nMarginals-1)), Matrix(I, nSize, nSize))
        else
            output[(i-1)*nSize+1: i*nSize, 1: end ] = constructFullMatrix(constructFullMatrix(ones(1, nSize^(i-1)), Matrix(I, nSize, nSize)), ones(1, nSize^(nMarginals-i)))
        end
    end
    return output
end


# Matrices for the Gangbo Swiech cost

function matrixVectorDifferenceBlock(col,nCol)
    output = spzeros( nCol - col +1, nCol - col)
    output[1,:] .= 1
    counter = 1
    for j = 1:nCol - col
        output[counter+1, j] = -1
        counter += 1
    end
    return output
end


function matrixVectorDifference(nCol)
    # maps [v1|...| vn] to [v1-v2| v1-v3 |...| v2-v3 |...|v{n-1}-vn]
    output = spzeros( nCol, div((nCol-1)*nCol, 2) )
    counter = 1
    row = 1
    for col = 1:nCol-1
        output[row:end , counter : counter + nCol-col -1] = matrixVectorDifferenceBlock(col,nCol)
        counter += nCol - col 
        row += 1

    end
    return output
end


function matrixVectorDifferenceDiagonal(nCol, dEuclid, dimension, matrixBlock)
    output = spzeros(div((nCol-1)*nCol, 2) * dEuclid * dimension , nCol * dEuclid * dimension)
    for i = 1: dEuclid * dimension
        output[ (i-1) * div((nCol-1)*nCol, 2) + 1: i * div((nCol-1)*nCol, 2) , (i-1) * nCol+ 1: i * nCol] = matrixBlock
    end
    return output
end


# ------------------------------------------------------------
# Proximal operators
# ------------------------------------------------------------


function proximalContinuityEquation( currentRho, currentMomentum, simP)
    # Compute the projection on the set of rho,momentum that satisfy the continuity equation

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]
    dimension = simP["dimension"]
    rhoShape = simP["rhoShape"]
    momentumShape = simP["momentumShape"]

    rhoIndex = (nTime+1)*nSpace^dimension
    momentumIndex = rhoIndex + nTime*dimension*nSpace^dimension


    # Compute the discrepancy
    discrepancy = simP["bigMatrixLC"] * vcat(vec(currentRho), vec(currentMomentum)) - simP["rhs"]
    # Invert the system
    discrepancy =  simP["choleskyBigMatrixLC"] \ discrepancy
    # Output
    output = vcat(vec(currentRho), vec(currentMomentum)) - simP["bigMatrixLC"]' * discrepancy
    return reshape(output[1:rhoIndex], rhoShape ), reshape(output[rhoIndex + 1: momentumIndex], momentumShape )

end


function proximalEnergyStar(a,b, tolDichotomy)

    # Returns zero of the function f(r) = (a-r) + 0.5 * |b|^2/(r+1)^2

    # If f(0) < tolDichotomy, return 0
    f = a + 0.5*norm(b)^2
    if f < tolDichotomy
        return 0.
    else

        # Newton's algorithm      
        rGuess = 0.
        counter = 0

        while abs(f) > tolDichotomy && counter <= 100
            f = a - rGuess + 0.5*norm(b)^2/(1+rGuess)^2
            df = -1 - norm(b)^2/(1+rGuess)^3
            rGuess -= f/df
            counter += 1
        end

        if counter == 101
            println("Warning: Newton fails to converge")
        end

        return rGuess

    end

end


function proximalEnergyVectorized(rho,momentum,simP)
    gamma = simP["gamma"]
    tolDichotomy = simP["tolDichotomy"]
    outputProj = zeros(size(rho))
    momentumVectors = reshape([@view momentum[Tuple(idx)..., :] for idx in CartesianIndices(size(rho))], size(rho) )
    outputProj = proximalEnergyStar.( rho/gamma, momentumVectors/gamma, tolDichotomy)

    return gamma*outputProj,
        outputProj .* momentum ./ (1 .+ outputProj)
end


# Big function to build the matrices

function constructMatrices(simP)

    # ------------------------------------------------------------
    # Extract values from the dictionary
    # ------------------------------------------------------------

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]
    dTotal = simP["dTotal"]
    dEuclid = simP["dEuclid"]
    deltaT = simP["deltaT"]
    deltaX = simP["deltaX"]
    nu = simP["nu"]
    regInversion = simP["regInversion"]
    nMarginals = simP["nMarginals"]
    baseMeasure = simP["baseMeasure"]
    marginals = simP["marginals"]



    # ------------------------------------------------------------
    # Defining auxiliary matrices
    # ------------------------------------------------------------

    # In time
    mDerivativeTime = matrixDerivativeLinear(nTime+1,deltaT) 
    mAverageTime = matrixAverageLinear(nTime+1) 

    # In space
    mAverageSpaceOneDimensional = matrixAverageLinearPer(nSpace) 
    mDerivativeSpace = matrixDerivativeLinearPer(nSpace, deltaX) 
    mDivergenceSpace = constructDimensionaFullMatrix(mDerivativeSpace, dTotal, "concatenated")
    mGradientSpace = -mDivergenceSpace' 
    mAverageSpace = constructDimensionaFullMatrix(mAverageSpaceOneDimensional, dTotal, "diagonal")  
    mLaplacianSpace = mDivergenceSpace * mGradientSpace 

    # Full matrices acting both on time and space 
    mAverageTimeFull = constructFullMatrix(mAverageTime,sparse(I, nSpace^dTotal, nSpace^dTotal)) 
    mDerivativeTimeFull = constructFullMatrix(mDerivativeTime,sparse(I, nSpace^dTotal, nSpace^dTotal)) 
    mAverageSpaceFull = constructFullMatrix(sparse(I,nTime,nTime), mAverageSpace) 
    mDivergenceSpaceFull = constructFullMatrix(sparse(I,nTime,nTime), mDivergenceSpace)  
    # For the Laplacian, both Laplacian in space and average in time
    mLaplacianSpaceFull = constructFullMatrix(mAverageTime, mLaplacianSpace) 


    # Boundary conditions
    mSelectBoundaryTimeInitial = spzeros(1,nTime+1)
    mSelectBoundaryTimeFinal = spzeros(1,nTime+1)
    mSelectBoundaryTimeInitial[1] = 1  
    mSelectBoundaryTimeFinal[end] = 1 
  
    mBoundaryConditionsInitialFull = constructFullMatrix(mSelectBoundaryTimeInitial,sparse(I,nSpace^dTotal, nSpace^dTotal)) 
    mBoundaryConditionsFinalFull = constructFullMatrix(mSelectBoundaryTimeFinal, matrixMarginalConstraints(nSpace^dEuclid, nMarginals))
   
    GangboSwiechMatrix = matrixVectorDifference(nMarginals)
    GangboSwiechMatrixDiagonal = matrixVectorDifferenceDiagonal(nMarginals, dEuclid, nSpace^dTotal, GangboSwiechMatrix')
   



    # ------------------------------------------------------------
    # Defining the big matrix for the continuity equation and boundary conditions
    # ------------------------------------------------------------
    nRowBigMatrix = nTime * nSpace^dTotal + (nSpace^dTotal + nMarginals * nSpace^dEuclid ) 
    nColumnBigMatrix = ((nTime+1) * nSpace^dTotal)+ (dTotal * nTime * nSpace^dTotal) 
   

    bigMatrixLC = spzeros( nRowBigMatrix, nColumnBigMatrix )

    # Derivatives of rho
    if nu > 1e-10
        # d rho/dt - nu/2 Laplacian rho
        bigMatrixLC[ 1:nTime*nSpace^dTotal, 1:(nTime+1)*nSpace^dTotal  ] = mDerivativeTimeFull - 0.5*nu*mLaplacianSpaceFull
    else
        # Only the temporal derivative
        bigMatrixLC[ 1:nTime*nSpace^dTotal, 1:(nTime+1)*nSpace^dTotal  ] = mDerivativeTimeFull
    end

    # Derivatives of momentum   
    bigMatrixLC[ 1:nTime*nSpace^dTotal, ((nTime+1)*nSpace^dTotal+1):((nTime+1)*nSpace^dTotal+nTime*dTotal*nSpace^dTotal) ] = mDivergenceSpaceFull

  

    # Boundary conditions
    bigMatrixLC[ nTime*nSpace^dTotal+1 : nTime*nSpace^dTotal+nSpace^dTotal, 1:(nTime+1)*nSpace^dTotal ] = mBoundaryConditionsInitialFull
    bigMatrixLC[ nTime*nSpace^dTotal+nSpace^dTotal+1 : nTime*nSpace^dTotal+nSpace^dTotal+nMarginals*nSpace^dEuclid, 1:(nTime+1)*nSpace^dTotal ] = mBoundaryConditionsFinalFull
    # Rhs of the continuity equation
    rhs = zeros(nRowBigMatrix)

    rhs[nTime*nSpace^dTotal+1 : nTime*nSpace^dTotal+nSpace^dTotal] = vec(baseMeasure)
    for i = 1:nMarginals
        rhs[(nTime*nSpace^dTotal+nSpace^dTotal)+(i-1)*(nSpace^dEuclid)+1 : (nTime*nSpace^dTotal+nSpace^dTotal)+i*(nSpace^dEuclid)] = vec(marginals[:,i])
    end

    # ------------------------------------------------------------
    # Cholesly factorizations
    # ------------------------------------------------------------

   
    choleskyAvgSpaceGangboSwiech = cholesky((GangboSwiechMatrixDiagonal * mAverageSpace) * (GangboSwiechMatrixDiagonal * mAverageSpace)' + sparse(I, size(GangboSwiechMatrixDiagonal * mAverageSpace, 1), size(GangboSwiechMatrixDiagonal * mAverageSpace, 1)))
    choleskyAvgSpace = cholesky(mAverageSpace*mAverageSpace' + sparse(I, dTotal*nSpace^dTotal, dTotal*nSpace^dTotal)) 
    choleskyAvgTime = cholesky(mAverageTime*mAverageTime' + sparse(I,nTime,nTime))
    choleskyBigMatrixLC = cholesky( bigMatrixLC * bigMatrixLC' + regInversion * sparse(I, nRowBigMatrix, nRowBigMatrix ))

    # Store the matrices

    simP["bigMatrixLC"] = bigMatrixLC
    simP["rhs"] = rhs
    simP["mAverageSpace"] = mAverageSpace
    simP["mAverageTime"] = mAverageTime
    simP["mAverageSpaceFull"] = mAverageSpaceFull
    simP["mAverageTimeFull"] = mAverageTimeFull
    simP["choleskyBigMatrixLC"] = choleskyBigMatrixLC
    simP["choleskyAvgSpace"] = choleskyAvgSpace
    simP["choleskyAvgTime"] = choleskyAvgTime
    simP["mLaplacianSpace"] = mLaplacianSpace
    simP["GangboSwiechMatrix"] = GangboSwiechMatrix
    simP["GangboSwiechMatrixDiagonal"] = GangboSwiechMatrixDiagonal
    simP["choleskyAvgSpaceGangboSwiech"] = choleskyAvgSpaceGangboSwiech
 
end

# RMMOT solver

function dynamical_RMMOT(
    nTime,
    nSpace,
    dTotal,
    dEuclid,
    nMarginals,
    # Boundary conditions
    marginals,
    baseMeasure;
    # Noise level
    nu::Float64 = 0.,
    nIter::Int64=1000,
    # Parameters for Primal-Dual 
    gamma = 1 / 85,
    theta = 0.0,
    tauScale =  0.5,
    # Tolerance and small parameters
    tolEnergy = 1e-10,
    tolDichotomy = 1e-10,
    maxIterDichotomy = 50,
    regInversion = 1e-10,
    # Displaying the time
    verbose = false
    )

    # Create the timer
    timer = TimerOutput()


    # ------------------------------------------------------------
    # Getting some information out of the arguments
    # ------------------------------------------------------------

    @timeit timer "Precomputations" begin

    # ------------------------------------------------------------
    # Defining some preliminary quantities
    # ------------------------------------------------------------

    gridTimeLarge = LinRange(0,1,nTime+1)
    deltaT = gridTimeLarge[2] - gridTimeLarge[1]

    gridSpaceCentered = LinRange(0,1-1/nSpace,nSpace)
    deltaX = gridSpaceCentered[2] - gridSpaceCentered[1]



    # Create tuples describing  shapes of arrays
     
    rhoShape = (nTime+1, ntuple(_ -> nSpace, dTotal)...)
    rhoAvgShape = (nTime, ntuple(_ -> nSpace, dTotal)...)
    momentumShape = (nTime, ntuple(_ -> nSpace, dTotal)..., dTotal)
    momentumEuclidShape = (nTime, ntuple(_ -> nSpace, dTotal)..., dEuclid, nMarginals)
    momentumDifferenceShape = (nTime, ntuple(_ -> nSpace, dTotal)..., dEuclid * div((nMarginals-1)*nMarginals, 2))
    momentumDifferenceEuclidShape = (nTime, ntuple(_ -> nSpace, dTotal)..., dEuclid, div((nMarginals-1)*nMarginals, 2) )



    # ------------------------------------------------------------
    # Initialize the objects
    # ------------------------------------------------------------

    # f^0
    rhoInterpolate = ones(rhoShape... ) / nSpace^dTotal 
    momentumInterpolate = zeros(momentumShape... )
    
    # g^0
    rhoAvg = ones(rhoAvgShape...) / nSpace^dTotal
    momentumDifference = zeros(momentumDifferenceEuclidShape... )

    # (h^0, h^1)
    rho = ones(2, rhoShape...) / nSpace^dTotal 
    momentum = zeros(2, momentumShape... )



    # The first component  corresponds to l'th iteratation and the second  to l+1
    rhoFirstSlice  = (1, ntuple(_ -> :, dTotal+1)...)
    rhoSecondSlice = (2, ntuple(_ -> :, dTotal+1)...)
    momentumFirstSlice = (1, ntuple(_ -> :, dTotal+2)...)
    momentumSecondSlice = (2, ntuple(_ -> :, dTotal+2)...)



    # ------------------------------------------------------------
    # Define big dictionary with all the fixed variables of the loop
    # ------------------------------------------------------------

    simP = Dict([("nu",nu),
        ("nIter",nIter),
        ("gamma",gamma),
        ("tolDichotomy",tolDichotomy),
        ("maxIterDichotomy",maxIterDichotomy),
        ("nTime",nTime),
        ("nSpace",nSpace),
        ("deltaT",deltaT),
        ("deltaX",deltaX),
        ("dTotal",dTotal),
        ("dimension",dTotal),
        ("regInversion",regInversion),
        ("rhoShape",rhoShape),
        ("rhoAvgShape",rhoAvgShape),
        ("momentumShape",momentumShape),
        ("momentumDifferenceEuclidShape",momentumDifferenceEuclidShape),
        ("nMarginals", nMarginals),
        ("dEuclid", dEuclid),
        ("baseMeasure", baseMeasure ),
        ("marginals", marginals)
        ])


    # ------------------------------------------------------------
    # Build the relevant derivation and averaging matrices
    # ------------------------------------------------------------

    constructMatrices(simP)

    mAverageTimeFull = simP["mAverageTimeFull"]
    mAverageSpaceFull = simP["mAverageSpaceFull"]
    GangboSwiechMatrix = simP["GangboSwiechMatrix"]


   

    # Compute norm of I
    block1 = mAverageTimeFull
    block2 = spzeros(size(mAverageTimeFull,1), size(mAverageSpaceFull,2))
    block3 = spzeros(size(mAverageSpaceFull,1), size(mAverageTimeFull,2))
    block4 = mAverageSpaceFull
    I = [block1 block2; block3 block4]
    u = rand(size(I,2))
    u = u / norm(u)
    normOld = 0.0
    normNew = norm(u)
    for _ in 1:50
        uNew = I' * (I * u)
        normNew = norm(uNew)
        u = uNew  / normNew
        if abs(normNew - normOld) < 1e-6
            break
        end
        normOld = normNew
    end
    normI  = sqrt(normNew)
    sigma = 1/gamma
    tau = tauScale  / (normI^2 * sigma)
    if verbose
        println("sigma*tau*normI^2 should be smaller than 1: ", sigma*tau*normI^2)
    end
    

     # End of the timer for initialization
    end

    
    # ------------------------------------------------------------
    # Primal-Dual loop
    # ------------------------------------------------------------

    @timeit timer "Loop" begin
        # Main loop
        for i = 1:nIter
            println("Iteration $(i) out of $(nIter)")

    
            # Last udpate: store the values to compute an error
            if i == nIter
                rhoC = copy(rho)
                momentumC = copy(momentum)
                rhoAvgC = copy(rhoAvg)
                momentumDifferenceC = copy(momentumDifference)
            end

            
            # g^{l+1}

            # map the momentum in f^l to the centered grid using the avaraging operator. Reshape it so that GangboSwiechMatrix can be applied
            momentumInterpolateCentered = reshape(mAverageSpaceFull * vec(momentumInterpolate), momentumShape...)
            momentumInterpolateCentered = reshape(momentumInterpolateCentered,  momentumEuclidShape)
            momentumInterpolateCentered = hcat(vec.(eachslice(momentumInterpolateCentered, dims=ndims(momentumInterpolateCentered)))...)

            momentumDifference = reshape(momentumDifference, momentumDifferenceShape)
            @timeit timer "Prox Energy" auxRBis, auxMBis = proximalEnergyVectorized((1/sigma).*(rhoAvg .+ sigma .* reshape(mAverageTimeFull * vec(rhoInterpolate), rhoAvgShape...)), (1/sigma).* (momentumDifference .+ sigma .* reshape( momentumInterpolateCentered * GangboSwiechMatrix , momentumDifferenceShape...)), simP)
            momentumDifference = reshape(momentumDifference, momentumDifferenceEuclidShape)
            auxMBis = reshape(auxMBis, momentumDifferenceEuclidShape)
            
            # Moreau’s identity
            rhoAvg = rhoAvg .+ sigma .* reshape(mAverageTimeFull * vec(rhoInterpolate), rhoAvgShape...) - sigma .* auxRBis  
            momentumDifference = momentumDifference .+ sigma .* reshape(momentumInterpolateCentered * GangboSwiechMatrix, momentumDifferenceEuclidShape...) - sigma .* auxMBis
            

            # h^{l+1}
            momentumDifference = hcat(vec.(eachslice(momentumDifference, dims=ndims(momentumDifference)))...)
            @timeit timer "Prox CE" auxR, auxM = proximalContinuityEquation( rho[rhoFirstSlice...] - tau .* reshape(mAverageTimeFull' * vec(rhoAvg), rhoShape...) ,   momentum[momentumFirstSlice...] - tau .* reshape(mAverageSpaceFull' * vec(reshape(momentumDifference * GangboSwiechMatrix', momentumShape...)), momentumShape...) , simP)
            momentumDifference = reshape(momentumDifference, momentumDifferenceEuclidShape...)
    
            rho[rhoSecondSlice...] = auxR  
            momentum[momentumSecondSlice...] = auxM 

            if verbose
                
                println("minimum of coupling is ", minimum(rho[rhoSecondSlice...]))

                # cost 
                vectorDifference = reshape(momentumInterpolateCentered * GangboSwiechMatrix, momentumDifferenceEuclidShape...)
                vectorDifference  = dropdims(vectorDifference ; dims = Tuple(findall(size(vectorDifference ) .== 1)))
                rhoAveraged = reshape(mAverageTimeFull * vec(rhoInterpolate), rhoAvgShape...)
                eps = 1e-8
                mask = rhoAveraged .> eps
                cost = sum(((vectorDifference.^2) ./ rhoAveraged) .* mask)
                println("cost is ", cost)

                # Error in the continuity equation
                errorContinuity = sum(abs.(simP["bigMatrixLC"] *  vcat(vec(rho[rhoSecondSlice...]), vec(momentum[momentumSecondSlice...])) - simP["rhs"]))
                println("error in continuity equation = ", errorContinuity)

            end
    
            # f^{l+1}       
            rhoInterpolate = rho[rhoSecondSlice...] + theta .* (rho[rhoSecondSlice...] - rho[rhoFirstSlice...]) 
            momentumInterpolate = momentum[momentumSecondSlice...] + theta .* (momentum[momentumSecondSlice...] - momentum[momentumFirstSlice...])

            # Update the h^l and h^{l+1} variables
            rho[rhoFirstSlice...] = rho[rhoSecondSlice...]
            momentum[momentumFirstSlice...] = momentum[momentumSecondSlice...]

            # Compute an error
            if i == nIter
                errorConsecutiveSteps = (sum(abs.( rho[rhoFirstSlice...] - rhoC[rhoFirstSlice...] )) + sum(abs.( momentum[momentumFirstSlice...] - momentumC[momentumFirstSlice...] )) + sum(abs.( rhoAvg[rhoFirstSlice...] - rhoAvgC[rhoFirstSlice...] )) + sum(abs.( momentumDifference[momentumFirstSlice...] - momentumDifferenceC[momentumFirstSlice...] )))/gamma
                errorContinuityEquation = sum(abs.(simP["bigMatrixLC"] *  vcat(vec(rho[rhoFirstSlice...]), vec(momentum[momentumFirstSlice...])) - simP["rhs"]))
                simP["errorConsecutiveSteps"] = errorConsecutiveSteps
                simP["errorContinuityEquation"] = errorContinuityEquation
            end
        end

    # End of the timer
    end

    if verbose
        display(timer)
    end

     # Return the output
    return rho[rhoSecondSlice...], rhoAvg, momentum[momentumSecondSlice...], simP
end
