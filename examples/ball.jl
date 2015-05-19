using DataFrames

include("../mdp/MDP.jl")
include("../mdp/GDP.jl")
include("../solvers/ball/Ball.jl")
include("../solvers/ball/Seq.jl")
include("../datatools/DataTools.jl")
include("../adpm/readParams.jl")
include("../adpm/predictAAR.jl")

using MDP, GDP, Seq, DataTools

# Airport and date we want to study
airport = :EWR # or :SFO
date = "2013-05-02"

# Get ADPM model for generating AARs
getAAR = predictAAR(readParams(airport),getForecast(airport))

# Cost of air delay/ground delay
ca = 2.

# Number of simulation time steps
nSteps = 27

# Random number generator
rng = MersenneTwister(rand(1:1000000))

# Read in AAR and schedule data
sm = getSchedule(airport,date)
aars = readcsv("../data/aars_"*string(airport)*".csv",Int16)
firstAAR = readcsv("../data/aar_first_"*string(airport)*".csv")

# Set up GDP parameters
gmp = GDPParams(int16(nSteps),:shortestfirst,sm,aars,firstAAR,ca,getAAR)

# Get the MDP generative model  
model = getGenerativeModel(gmp)

# Get policy for Ball model
policy = getBallPolicy(gmp,rng)

# Simulate
reward = simulate(model,[],policy,nSteps,rng)

# Print statement
println("Sum of rewards for Ball model at "*string(airport))
println(reward)
