using DataFrames

include("../mdp/MDP.jl")
include("../mdp/GDP.jl")
include("../solvers/adp/ADP.jl")
include("../datatools/DataTools.jl")
include("../adpm/readParams.jl")
include("../adpm//predictAAR.jl")

using MDP, GDP, ADP, DataTools

# Airport and date we want to study
airport = :EWR # or :SFO
date = "2013-05-02"

# Get ADPM model for generating AARs
getAAR = predictAAR(readParams(airport),getForecast(airport))

# Cost of air delay/ground delay
ca = 2.

# Number of simulation time steps
nSteps = 27

# Planning horizon length (hours)
phl = 2

# Random number generator
rng = MersenneTwister(rand(1:1000000))

# Read in AAR and schedule data
sm = getSchedule(airport,date)
aars = readcsv("../data/aars_"*string(airport)*".csv",Int16)
firstAAR = readcsv("../data/aar_first_"*string(airport)*".csv")

# Set up GDP parameters
gmp = GDPParams(int16(phl),:shortestfirst,sm,aars,firstAAR,ca,getAAR)

# Get the MDP generative model  
model = getGenerativeModel(gmp)

# Get GDP-specific functions needed for ADP policy 
getAction, getPossibleActions = getPolicyFunctions(gmp)

# Set up ADP solution parameters
d = int16(6) # depth
ec = 1000. # exploration constant
p = ADPParams(d,ec,rng,getPossibleActions,getAction,model.getNextState,model.getReward)

# Get policy for ADP
policy = selectAction

# Simulate
reward = simulate(model,p,policy,nSteps,rng)

# Print statement
println("Sum of rewards for Approximate Dynamic Programming at "*string(airport))
println(reward)
