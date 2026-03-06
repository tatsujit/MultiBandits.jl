# using Pkg
# Pkg.add(["Random", "DataFrames", "CSV", "Distributions", "StatsBase", "Statistics", 
#          "Optim", "Distributed", "ProgressMeter", "YAML", "SpecialFunctions"])

using Random, DataFrames, CSV # for data manipulation and CSV handling
using Distributions, StatsBase # for statistical distributions and functions
using Distributed # for parallel computing
# _utils.jl must be loaded first (defines @ic0 macro used in core files)
include("_utils.jl")

# path = "src/"
path = "core/"
include(path * "action_value_estimator.jl")
include(path * "policy.jl")
include(path * "agent.jl")
include(path * "environment.jl")
include(path * "history.jl")
include(path * "system.jl")
include(path * "evaluation.jl")
