#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Reexport
@reexport using Flux
#using Markdown: MD

export make, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

"""

"""
make

export spread
include("utilities.jl")

export cyanotype
include("cyanotype.jl")

export NoNormBp, BatchNormBp, GroupNormBp, InstanceNormBp
include("norm.jl")

export ConvBp, DoubleConvBp, NConvBp, HybridAtrouConvBp
include("conv.jl")

end
