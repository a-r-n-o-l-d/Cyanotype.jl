#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    - BpDoubleConv with kwargs
    x make without kwargs
    - cleaning
    x replace NoNorm by nothing
    - ResidualConnection with dropout
    x volmetric -> volume (more clear)
    - spread => broadcast(?)
    x activation identity by default
    x BpDConv => BpDoubleConv
=#
module Cyanotype

using Reexport
@reexport using Flux
using Flux: zeros32, ones32, glorot_uniform
using Statistics

export make, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

"""

"""
make

make(::Nothing) = identity

export spread, flatten_layers, cyanotype, KwargsMapping, @cyanotype, BpBatchNorm,
       BpGroupNorm, BpInstanceNorm, BpConv, BpDoubleConv, BpNConv, BpPointwiseConv,
       BpChannelExpansionConv, BpDepthwiseConv, BpHybridAtrouConv, BpSqueezeExcitation,
       BpMBConv, BpPixelClassifier, BpChannelAttention, BpSpatialAttention, BpCBAM

include("utilities.jl")

include("cyanotype.jl")

include("units/normalizations.jl")

include("units/convolutions.jl")

export BpMaxDown, BpMeanDown, BpNearestUp, BpLinearUp, BpConvTransposeUp, BpPixelShuffleUp
include("units/samplers.jl")

export uchain, BpUEncoder, BpUDecoder, BpUBridge, BpUNet
include("models/uchain.jl")
include("models/unet.jl")

end
