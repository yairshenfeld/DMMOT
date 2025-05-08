
using Roots
using QuadGK

# ----------------------------------------
#  One-Dimensional distributions 
# ---------------------------------------


# ------------------------------
#  PDF, CDF, and inverse CDF of the uniform distribution on [0,1]
# ------------------------------

"""
Uniform PDF on [0,1].

f(x) = 1, 0 ≤ x ≤ 1.
"""
function uniformPdf(x::Number)
    return 1.0
end

function uniformPdf(x::AbstractArray)
    return ones(Float64, size(x))
end

"""
CDF for the uniform distribution.

F(x) = x, 0 ≤ x ≤ 1.
"""
function uniformCdf(x::Number)
    return x
end

function uniformCdf(x::AbstractArray)
    return x  
end

"""
Inverse CDF for the uniform distribution.

F(x) = x, 0 ≤ x ≤ 1.
"""
function uniformInvCdf(u::Number)
    return u
end

function uniformInvCdf(u::AbstractArray)
    return u
end


# ------------------------------
#  PDF, CDF, and inverse CDF of the camel distribution on [0,1]
# ------------------------------

"""
Piecewise-linear PDF on [0,1] with two humps at x = 0.25 and x = 0.75.

f(x) =
  8x,                0 ≤ x < 0.25  
  4 − 8x,            0.25 ≤ x < 0.5  
  8x − 4,            0.5 ≤ x < 0.75  
  8 − 8x,            0.75 ≤ x ≤ 1  
  0,                 otherwise.
"""
function camelPdf(x::Number)
    if x ≥ 0 && x < 0.25
        return 8.0 * x
    elseif x ≥ 0.25 && x < 0.5
        return 4.0 - 8.0 * x
    elseif x ≥ 0.5 && x < 0.75
        return 8.0 * x - 4.0
    elseif x ≥ 0.75 && x ≤ 1
        return 8.0 - 8.0 * x
    else
        return 0.0
    end
end

function camelPdf(x::AbstractArray)
    return [camelPdf(xi) for xi in x]
end

"""
CDF for the piecewise-linear “double-hump” distribution.
F(x) is defined piecewise as quadratic functions.
"""
function camelCdf(x::Number)
    if x ≤ 0
        return 0.0
    elseif x < 0.25
        return 4.0 * x^2
    elseif x < 0.5
        return -4.0 * x^2 + 4.0 * x - 0.5
    elseif x < 0.75
        return 4.0 * x^2 - 4.0 * x + 1.5
    elseif x ≤ 1
        return -4.0 * x^2 + 8.0 * x - 3.0
    else
        return 1.0
    end
end

function camelCdf(x::AbstractArray)
    return [camelCdf(xi) for xi in x]
end

"""
Inverse CDF (quantile function) for the double-hump distribution.
Based on piecewise formulas derived from F(x).
"""
function camelInvCdf(u::Number)
    if u ≤ 0
        return 0.0
    elseif u ≥ 1
        return 1.0
    elseif u < 0.25
        # For x in [0, 0.25]: F(x) = 4x²  ⟹  x = sqrt(u)/2
        return 0.5 * sqrt(u)
    elseif u < 0.5
        # For x in [0.25, 0.5]: Solve 4x² − 4x + (u + 0.5) = 0.
        A = 4.0; B = -4.0; C = u + 0.5
        disc = B^2 - 4.0*A*C
        xplus = (-B + sqrt(disc))/(2.0*A)
        xminus = (-B - sqrt(disc))/(2.0*A)
        return min(xplus, xminus)
    elseif u < 0.75
        # For x in [0.5, 0.75]: Solve 4x² − 4x + (1.5 − u) = 0.
        A = 4.0; B = -4.0; C = 1.5 - u
        disc = B^2 - 4.0*A*C
        xplus = (-B + sqrt(disc))/(2.0*A)
        xminus = (-B - sqrt(disc))/(2.0*A)
        return max(xplus, xminus)
    else
        # For x in [0.75, 1]: Solve 4x² − 8x + (u + 3) = 0.
        A = 4.0; B = -8.0; C = u + 3.0
        disc = B^2 - 4.0*A*C
        xplus = (-B + sqrt(disc))/(2.0*A)
        xminus = (-B - sqrt(disc))/(2.0*A)
        return min(xplus, xminus)
    end
end

function camelInvCdf(u::AbstractArray)
    return [camelInvCdf(ui) for ui in u]
end



# ------------------------------
#  PDF, CDF, and inverse CDF of the tent distribution on [0,1]
# ------------------------------

"""
Tent (triangular) PDF on [0,1] with a peak at x = 0.5.

f(x) =
   4x           for 0 ≤ x ≤ 0.5,
   4(1 − x)     for 0.5 < x ≤ 1,
   0            otherwise.
"""
function tentPdf(x::Number)
    if x ≥ 0 && x ≤ 0.5
        return 4.0 * x
    elseif x > 0.5 && x ≤ 1.0
        return 4.0 * (1.0 - x)
    else
        return 0.0
    end
end

function tentPdf(x::AbstractArray)
    return [tentPdf(xi) for xi in x]
end

"""
CDF for the tent distribution on [0,1]:

F(x) =
   0                    for x < 0,
   2x²                  for 0 ≤ x ≤ 0.5,
   1 − 2(1 − x)²        for 0.5 < x ≤ 1,
   1                    for x > 1.
"""
function tentCdf(x::Number)
    if x ≤ 0
        return 0.0
    elseif x ≤ 0.5
        return 2.0 * x^2
    elseif x ≤ 1.0
        return 1.0 - 2.0 * (1.0 - x)^2
    else
        return 1.0
    end
end

function tentCdf(x::AbstractArray)
    return [tentCdf(xi) for xi in x]
end

"""
Inverse CDF (quantile function) for the tent distribution.
For u in [0,1]:

For u in [0, 0.5]:
   Solve 2x² = u  ⟹  x = √(u/2).
For u in [0.5, 1]:
   Solve 1 − 2(1 − x)² = u  ⟹  x = 1 − √((1 − u)/2).
"""
function tentInvCdf(u::Number)
    if u ≤ 0.0
        return 0.0
    elseif u ≥ 1.0
        return 1.0
    elseif u ≤ 0.5
        return sqrt(u/2.0)
    else
        return 1.0 - sqrt((1.0 - u)/2.0)
    end
end

function tentInvCdf(u::AbstractArray)
    return [tentInvCdf(ui) for ui in u]
end


# ------------------------------
#  PDF, CDF, and inverse CDF of the sine distribution on [0,1]
# ------------------------------

function sinePdf(x::Number)
    if 0.0 ≤ x ≤ 1.0
        return (π / 2) * sin(π * x)
    else
        return 0.0
    end
end

function sinePdf(x::AbstractArray)
    return [sinePdf(xi) for xi in x]
end

"""
CDF for sine-based distribution:
    F(x) = (1 - cos(πx)) / 2
"""
function sineCdf(x::Number)
    if x ≤ 0.0
        return 0.0
    elseif x ≥ 1.0
        return 1.0
    else
        return (1.0 - cos(π * x)) / 2.0
    end
end

function sineCdf(x::AbstractArray)
    return [sineCdf(xi) for xi in x]
end

"""
Inverse CDF:
    u ∈ [0,1] ⟹ x = acos(1 - 2u) / π
"""
function sineInvCdf(u::Number)
    if u ≤ 0.0
        return 0.0
    elseif u ≥ 1.0
        return 1.0
    else
        return acos(1.0 - 2.0 * u) / π
    end
end

function sineInvCdf(u::AbstractArray)
    return [sineInvCdf(ui) for ui in u]
end



#------------------------------------------------------------
#  Numerically computes  CDF and inverse CDF 
#------------------------------------------------------------


function computeInverseCDF(cdf; tol=1e-8)
    function invCdf(x)
        @assert 0.0 ≤ x ≤ 1.0 "x must be in [0,1]"
        f(y) = cdf(y) - x
        find_zero(f, (0.0, 1.0), Bisection(), atol=tol)
    end
    return invCdf
end


function computeCdf(f::Function)
    return x -> x < 0 ? 0.0 :
                x > 1 ? 1.0 :
                quadgk(f, 0, x)[1]
end

#------------------------------------------------------------
#  Transport maps computations  
#------------------------------------------------------------


function transportMapFromJointLaw(nSpace, dEuclid, coupling, marginalSource, marginalSourceIdx, marginalTargetIdx)

    output = spzeros((ntuple(_ -> nSpace, dEuclid)..., dEuclid))

    # Margenalize the coupling to get the joint distribution of marginalSource and marginalTarget
    
    marginalSourceIndices = ((marginalSourceIdx-1) * dEuclid)+ 1: ((marginalSourceIdx-1) * dEuclid) + dEuclid
    marginalTargetIndices = ((marginalTargetIdx-1) * dEuclid)+ 1: ((marginalTargetIdx-1) * dEuclid) + dEuclid
    dimsToSum = tuple(setdiff(1:ndims(coupling), union(marginalSourceIndices, marginalTargetIndices))...)
    jointTwoMarginals = dropdims(sum(coupling, dims = dimsToSum ), dims = dimsToSum)
    jointTwoMarginals = reshape(jointTwoMarginals, nSpace, nSpace)


    positions = LinRange(0/nSpace, 1-1/(nSpace), nSpace)
    targetDim = marginalTargetIdx < marginalSourceIdx ? 1 : 2
    positions = exp.(2π*1im .* positions)  # maps coordinates fro [0,1] to the unit circle
    positionsArrays = [reshape(positions, ntuple(j -> j == targetDim ? nSpace : 1, 2)...) for i in 1:dEuclid]
    if marginalSourceIdx < marginalTargetIdx
        sumDims = (dEuclid+1):(2*dEuclid)
    elseif marginalSourceIdx > marginalTargetIdx  
        sumDims = 1:dEuclid
    else 
        error("marginalSourceIdx and marginalTargetIdx must be different")
    end
    outputComponents = [ dropdims(sum(jointTwoMarginals .* positionsArrays[i], dims = sumDims); dims = Tuple(sumDims)) ./ marginalSource for i in 1:dEuclid ]
    output = cat(outputComponents...; dims = dEuclid+1)
    complexPositions = vec(output)
    transportMap = angle.(complexPositions) ./ (2π)  # map coordinates from the unit circle to [0,1]
    transportMap .= ifelse.(transportMap .< 0.0, transportMap .+ 1.0, transportMap)
    return transportMap
end



