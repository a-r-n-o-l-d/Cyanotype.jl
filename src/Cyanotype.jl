#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    - tool to check if AbstractConvBp (and fields) is volumetric
    - spread avec type de blueprint sur lesquel on change une options : spread(bp, ConvBp; activation=swish)
    - docs, docs, docs, docs
    x renommage : mettre Bp en fin de nom
    - showing blueprints is ugly => GarishPrint or custom dump ? cyanotype _show_func
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
    - check if we can use @cyanotype outside the package
=#
module Cyanotype

using Reexport
@reexport using Flux
using Flux: zeros32, ones32, glorot_uniform, flatten
using ChainRules
using Statistics

export make, spread, flatten_layers, cyanotype, KwargsMapping, @cyanotype, BatchNormBp,
       GroupNormBp, InstanceNormBp, ConvBp, DoubleConvBp, NConvBp, PointwiseConvBp,
       ChannelExpansionConvBp, DepthwiseConvBp, HybridAtrouConvBp, SqueezeExcitationBp,
       MbConvBp, PixelClassifierBp, ChannelAttentionBp, SpatialAttentionBp, CBAMBp,
       FusedMbConvBp, LabelClassifierBp, MaxDownsamplerBp, MeanDownsamplerBp,
       NearestUpsamplerBp, LinearUpsamplerBp, ConvTransposeUpsamplerBp,
       PixelShuffleUpsamplerBp, uchain, UEncoderBp, UDecoderBp, UBridgeBp, UNetBp,
       EfficientNetStageBp, EfficientNetBp, PixelMapBp, EfficientUNetBp, ResCBAMBp,
       ResidualConvBp, AxialDWConvBp

const CyFloat = Union{Float16, Float32, Float64}

abstract type AbstractBlueprint end

"""

"""
make

make(::Nothing) = identity

make(::Nothing, ::Any) = identity

make(::Nothing, ::Any, ::Any) = identity

include("utilities.jl")

include("cyanotype.jl")

include("units/normalizations.jl")

include("units/convolutions.jl")

include("units/classifiers.jl")

include("units/pixmap.jl")

include("units/samplers.jl")

include("models/unets/unet.jl")

include("models/efficientnets/efficientnet.jl")

include("models/unets/effunet.jl")

end
