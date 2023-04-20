#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    - docs, docs, docs, docs
    - showing blueprints is ugly => GarishPrint or custom ?
    - BpMBConv => BpMbConv
    - BpUDecoder => dispatch on make not on _make
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

make(::Nothing, ::Any) = identity

export spread, flatten_layers, cyanotype, KwargsMapping, @cyanotype, BpBatchNorm,
       BpGroupNorm, BpInstanceNorm, BpConv, BpDoubleConv, BpNConv, BpPointwiseConv,
       BpChannelExpansionConv, BpDepthwiseConv, BpHybridAtrouConv, BpSqueezeExcitation,
       BpMBConv, BpPixelClassifier, BpChannelAttention, BpSpatialAttention, BpCBAM,
       BpFusedMBConv

include("utilities.jl")

include("cyanotype.jl")

include("units/normalizations.jl")

include("units/convolutions.jl")

export BpMaxDownsampler, BpMeanDownsampler, BpNearestUpsamplers, BpLinearUpsampler, BpConvTransposeUpsampler, BpPixelShuffleUpsampler
include("units/samplers.jl")

export uchain, BpUEncoder, BpUDecoder, BpUBridge, BpUNet
include("models/uchain.jl")
include("models/unet.jl")

end
