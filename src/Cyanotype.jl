#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Flux
using Markdown: MD

export CyanoFloat

const CyanoFloat = Union{Float16, Float32, Float64}

include("utilities.jl")

#include("kwmapping.jl")
include("cyano.jl")
#export register_mapping!
#include("config.jl")

export CyanoIdentityNorm, CyanoBatchNorm, CyanoGroupNorm
include("norm.jl")

end
