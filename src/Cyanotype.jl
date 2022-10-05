#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Flux
using Markdown: MD

export build, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractCyanotype end #AbstrctCyanotypeBlueprint

export spread
include("utilities.jl")

export cyanotype
include("cyanotype.jl")

#include("kwmapping.jl")
#include("cyano.jl")
#export register_mapping!
#include("config.jl")

export CyNoNorm, CyBatchNorm, CyGroupNorm, CyInstanceNorm
include("norm.jl")

export CyConv, CyDoubleConv, CyNConv, CyHybridAtrouConv
include("conv.jl")

end
