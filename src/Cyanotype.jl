#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    x make without kwargs
    - cleaning
    x replace NoNorm by nothing
    - ResidualConnection with dropout
    x volmetric -> volume (more clear)
    - spread => broadcast(?)
    x activation identity by default
    - BpDConv => BpDoubleConv
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

make(::Nothing) = identity

export spread, flatten_layers
include("utilities.jl")

export cyanotype, KwargsMapping, @cyanotype
include("cyanotype.jl")

export BpNoNorm, BpBatchNorm, BpGroupNorm, BpInstanceNorm
include("units/normalizations.jl")

export BpConv, BpDoubleConv, BpNConv, BpPointwiseConv, BpChannelExpansionConv, BpDepthwiseConv,
       BpHybridAtrouConv, BpSqueezeExcitation, BpMBConv
include("units/convolutions.jl")

export BpMaxDown, BpMeanDown, BpNearestUp, BpLinearUp, BpConvTransposeUp, BpPixelShuffleUp
include("units/samplers.jl")

include("units/classifiers.jl")

#include("u_network.jl")

end
