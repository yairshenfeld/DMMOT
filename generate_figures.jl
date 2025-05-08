# Generate the figures of the paper. 


include("dynamical_RMMOT.jl")
include("functions.jl")
using Base.Iterators: product
using Plots
pyplot()

script_dir = dirname(@__FILE__)  # Get the directory of the current script
cd(script_dir)



# ------------------------------------------------------------
# 3 marginals in dimension 1 
# ------------------------------------------------------------



# Dimension of space where marginals are defined
dEuclid = 1 
# Number of marginals
nMarginals = 3
# regularization parameter
nu = 0.00
# Size of grids
nTime = 10
nSpace = 10

# number of iterations
nIter = 5000  # 5000

dTotal = nMarginals * dEuclid # toal dimension


# ------------------------------------------------------------
# Parameters for Primal-Dual method
# ------------------------------------------------------------

gamma = 1 / 85 # gamma = 1/sigma  = 1 / 85 
theta = 1
tauScale = 0.1 # If tauScale is too big the algorithm diverges


# ------------------------------------------------------------
# Define the marginals
# ----------------------------------------------------------

positions = LinRange(0/nSpace, 1-1/(nSpace), nSpace) 

# base measure is always uniform
baseMeasureAux = uniformPdf(positions)
baseMeasure = zeros(Float64, ntuple(_ -> nSpace, dTotal))
indices = [(i -> ntuple(_ -> i, dTotal))(i) for i in 1:nSpace]
baseMeasure[CartesianIndex.(indices)] .= baseMeasureAux
baseMeasure = baseMeasure / sum(baseMeasure)


marginalOne = sinePdf(positions)
marginalTwo = camelPdf(positions)
marginalThree = tentPdf(positions)


# Avoid division by zero
minDensityOne = 0.2
minDensityTwo = 0
minDensityThree = 0
marginalOne = marginalOne .+ minDensityOne
marginalTwo = marginalTwo .+ minDensityTwo
marginalThree = marginalThree .+ minDensityThree


# Noramlize the densities
marginalOne = marginalOne / sum(marginalOne)
marginalTwo = marginalTwo / sum(marginalTwo)
marginalThree = marginalThree / sum(marginalThree)


# Correct the CDFs because of the addition of minDensity

marginalOneCdf = x -> (sineCdf(x) .+ minDensityOne * x) .* (1/(1+ minDensityOne))
marginalTwoCdf = x -> (camelCdf(x) .+ minDensityTwo * x) .* (1/(1+ minDensityTwo))
marginalThreeCdf = x -> (tentCdf(x) .+ minDensityThree * x) .* (1/(1+ minDensityThree))

# Compute inverse CDFs
marginalTwoInvCdf = computeInverseCDF(marginalTwoCdf)
marginalThreeInvCdf = computeInverseCDF(marginalThreeCdf)

# First transport map: sine to camel (marginal one to marginal two)
marginalFirstSource = marginalOne
marginalFirstTarget = marginalTwo
marginalFirstSourceIdx = 1
marginalFirstTargetIdx = 2

# First transport map: sine to tent (marginal one to marginal three)
marginalSecondSource = marginalOne
marginalSecondTarget = marginalThree
marginalSecondSourceIdx = 1
marginalSecondTargetIdx = 3



# Compute CDFs and inverse CDFs for computation of analytic transport maps
marginalFirstSourceCdf = marginalOneCdf
marginalFirstTargetInvCdf = marginalTwoInvCdf

marginalSecondSourceCdf = marginalOneCdf
marginalSecondTargetInvCdf = marginalThreeInvCdf



# Populate marginals 

marginals = zeros(nSpace^dEuclid, nMarginals)
marginals[:, 1] = marginalOne
marginals[:, 2] = marginalTwo
marginals[:, 3] = marginalThree

# # ------------------------------------------------------------
# # Compute the solution
# # ------------------------------------------------------------


println("start")

rho, rhoAvg, momentum, simP = dynamical_RMMOT(nTime, nSpace, dTotal, dEuclid, nMarginals , marginals, baseMeasure;
    nu = nu,
    nIter = nIter,
    gamma = gamma,
    theta = theta,
    tauScale = tauScale,
    verbose = false)

# Plot the error made

println("Final error consecutive steps")
println(simP["errorConsecutiveSteps"])
println("Discrepancy continuity equation")
println(simP["errorContinuityEquation"])


#------------------------------------------------------------
#  Compute transport maps   
#------------------------------------------------------------


# Extract the coupling from the output of dynamical_RMMOT
jointLawFinal = rho[nTime+1, ntuple(_ -> :, dTotal)...]

positionsExact = LinRange(0/500, 1-1/(500), 500)


# Compute the first transport map analytically and numerically
transportFirstSourceToFirstTarget = transportMapFromJointLaw(nSpace, dEuclid, jointLawFinal, marginalFirstSource, marginalFirstSourceIdx, marginalFirstTargetIdx)
analyticTransportFirstSourceToFirstTarget  = marginalFirstTargetInvCdf.(marginalFirstSourceCdf.(positionsExact))

# Compute the second transport map analytically and numerically
transportSecondSourceToSecondTarget = transportMapFromJointLaw(nSpace, dEuclid, jointLawFinal, marginalSecondSource, marginalSecondSourceIdx, marginalSecondTargetIdx)
analyticTransportSecondSourceToSecondTarget  = marginalSecondTargetInvCdf.(marginalSecondSourceCdf.(positionsExact))


# ------------------------------------------------------------
# Plot and save results
# ------------------------------------------------------------

# Plot the marginals


marginalOneExact = sinePdf(positionsExact)
marginalTwoExact = camelPdf(positionsExact)
marginalThreeExact = tentPdf(positionsExact)

# Avoid division by zero
minDensityOne = 0.2
minDensityTwo = 0
minDensityThree = 0
marginalOneExact = marginalOneExact .+ minDensityOne
marginalTwoExact = marginalTwoExact .+ minDensityTwo
marginalThreeExact = marginalThreeExact .+ minDensityThree


# Noramlize the densities
marginalOneExact = marginalOneExact / sum(marginalOneExact)
marginalTwoExact = marginalTwoExact / sum(marginalTwoExact)
marginalThreeExact = marginalThreeExact / sum(marginalThreeExact)


plotMarginalOne = plot(positionsExact, marginalOneExact;
    color = :black,
    linewidth = 2,
    title = "First marginal (mu1)",
    legend = false)
  
savefig(plotMarginalOne, "mu1.pdf")

plotMarginalTwo = plot(positionsExact, marginalTwoExact;
    color = :black,
    linewidth = 2,
    title = "Second marginal (mu2)",
    legend = false)
  
savefig(plotMarginalTwo, "mu2.pdf")


plotMarginalThree = plot(positionsExact, marginalThreeExact;
    color = :black,
    linewidth = 2,
    title = "Third marginal (mu3)",
    legend = false)
  
savefig(plotMarginalThree, "mu3.pdf")


# Plot the transport maps 


# Transport map between marginal one to marginal two 
plotFirstTranportMaps = plot(positionsExact, analyticTransportFirstSourceToFirstTarget;
    label = "exact",
    color = :red,
    linewidth = 2,
    title = "Transport maps between first and second marginals",
    legend = :topleft)

plot!(plotFirstTranportMaps, positions, transportFirstSourceToFirstTarget;
    label = "numerical",
    seriestype = :scatter,
    marker = (:circle, 4, 0.7),
    markercolor = :blue,
    linecolor = :transparent,
    legend = :topleft)

savefig(plotFirstTranportMaps, "transport_maps_mu1_to_mu2.pdf")


# Transport map between marginal one to marginal three
plotSecondTranportMaps = plot(positionsExact, analyticTransportSecondSourceToSecondTarget;
    label = "exact",
    color = :red,
    linewidth = 2,
    title = "Transport maps between first and third marginals",
    legend = :topleft)

plot!(plotSecondTranportMaps, positions, transportSecondSourceToSecondTarget;
    label = "numerical",
    seriestype = :scatter,
    marker = (:circle, 4, 0.7),
    markercolor = :blue,
    linecolor = :transparent,
    legend = :topleft)

savefig(plotSecondTranportMaps, "transport_maps_mu1_to_mu3.pdf")


# Combine plots

layoutMarginals = @layout [a{0.30w} b{0.30w} c{0.30w}]
combinedMargianlPlots = plot(
plotMarginalOne, plotMarginalTwo, plotMarginalThree;
  layout = layoutMarginals,
  size   = (1200, 400),   
  framestyle = :box       
)
savefig(combinedMargianlPlots, "all_marginals.pdf")


layoutTransportMaps = @layout [a{0.45w} b{0.45w}]
combinedTransportMapsPlots = combinedMargianlPlots = plot(plotFirstTranportMaps, plotSecondTranportMaps; layout = layoutTransportMaps, size = (800, 400), framestyle = :box, titlefont  = font(8) )
savefig(combinedTransportMapsPlots, "all_transport_maps.pdf")



