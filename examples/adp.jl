using DataFrames

include("../mdp/MDP.jl")
include("../mdp/GDP.jl")
include("../other/Auxfuncs.jl")
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
nSteps = 24

# Planning horizon length (hours)
phl = 2

# Random number generator
rng = MersenneTwister(rand(1:1000000))

# Read in AAR and schedule data
sm = getSchedule(airport,date)
aars = convert(Array{Int16},readcsv("../data/aars_"*string(airport)*".csv",Float64))
firstAAR = readcsv("../data/aar_first_"*string(airport)*".csv")

# Get the AARs used as actions in the ADP model
aars_adp = convert(Array{Int16},readcsv("../data/aars_"*string(airport)*"_adp.csv",Float64))

# CDM model - :shortestfirst, :longestfirst, or :random
cdmModel = :shortestfirst

# Set up GDP parameters
gmp = GDPParams(Int16(phl),cdmModel,sm,aars,firstAAR,ca,getAAR)
gmp_adp = GDPParams(Int16(phl),cdmModel,sm,aars_adp,firstAAR,ca,getAAR)

# Get the MDP generative model  
model = getGenerativeModel(gmp)
model_adp = getGenerativeModelADP(gmp)

# Get GDP-specific functions needed for ADP policy 
getAction, getPossibleActions = getPolicyFunctions(gmp_adp)

# Set up ADP solution parameters
d = Int16(6) # depth
ec = 1000. # exploration constant
p = ADPParams(d,ec,rng,getPossibleActions,getAction,model_adp.getNextState,model.getReward)

# Get policy for ADP
policy = selectAction

# Simulate
reward = simulate(model,p,policy,nSteps,rng)

# Print statement
println("Sum of rewards for Approximate Dynamic Programming at "*string(airport))
println(reward)   
