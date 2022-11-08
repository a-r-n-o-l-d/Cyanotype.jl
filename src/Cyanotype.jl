#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Reexport
@reexport using Flux
using Flux: zeros32, ones32, glorot_uniform

export make, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

"""

"""
make

export spread, flatten_layers
include("utilities.jl")

export cyanotype, KwargsMapping, @cyanotype
include("cyanotype.jl")

export NoNormBp, BatchNormBp, GroupNormBp, InstanceNormBp
include("norm.jl")

export ConvBp, DoubleConvBp, NConvBp, HybridAtrouConvBp
include("conv.jl")

export SqueezeExciteBp
include("squeeze_excite.jl")

end
