#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    - docs, docs, docs, docs
    - renommage : mettre Bp en fin de nom
    - showing blueprints is ugly => GarishPrint or custom dump ?
    - BpMBConv => BpMbConv
    - UDecoderBp => dispatch on make not on _make
    - DoubleConvBp with kwargs
    x make without kwargs
    - cleaning
    x replace NoNorm by nothing
    - ResidualConnection with dropout
    x volmetric -> volume (more clear)
    - spread => broadcast(?)
    x activation identity by default
    x BpDConv => DoubleConvBp
=#
module Cyanotype

using Reexport
@reexport using Flux
using Flux: zeros32, ones32, glorot_uniform, flatten
using Statistics

export make, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

"""

"""
make

make(::Nothing) = identity

make(::Nothing, ::Any) = identity

export spread, flatten_layers, cyanotype, KwargsMapping, @cyanotype, BatchNormBp,
       GroupNormBp, InstanceNormBp, ConvBp, DoubleConvBp, NConvBp, PointwiseConvBp,
       ChannelExpansionConvBp, DepthwiseConvBp, HybridAtrouConvBp, SqueezeExcitationBp,
       MbConvBp, PixelClassifierBp, ChannelAttentionBp, SpatialAttentionBp, CBAMBp,
       FusedMbConvBp, LabelClassifierBp

include("utilities.jl")

include("cyanotype.jl")

include("units/normalizations.jl")

include("units/convolutions.jl")

include("units/classifiers.jl")

export MaxDownsamplerBp, MeanDownsamplerBp, NearestUpsamplerBp, LinearUpsamplerBp, ConvTransposeUpsamplerBp, PixelShuffleUpsamplerBp
include("units/samplers.jl")

export uchain, UEncoderBp, UDecoderBp, UBridgeBp, UNetBp

include("models/unets/unet.jl")

end
