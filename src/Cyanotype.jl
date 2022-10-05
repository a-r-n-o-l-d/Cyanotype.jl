#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Flux
using Markdown: MD

export make, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

export spread
include("utilities.jl")

export cyanotype
include("cyanotype.jl")

#include("kwmapping.jl")
#include("cyano.jl")
#export register_mapping!
#include("config.jl")

export NoNormBp, BatchNormBp, GroupNormBp, InstanceNormBp
include("norm.jl")

export ConvBp, DoubleConvBp, NConvBp, HybridAtrouConvBp
include("conv.jl")

end
