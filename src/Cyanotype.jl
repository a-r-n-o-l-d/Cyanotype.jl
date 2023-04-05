#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    x make without kwargs
    - cleaning
    x replace NoNorm by nothing
    - ResidualConnection with dropout
    x volmetric -> volume (more clear)
    - spread => broadcast
    - activation identity by default
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

export BpNoNorm, BpBatchNorm, BpGroupNorm, BpInstanceNorm
include("normalizations.jl")

export BpConv, BpDConv, BpNConv, BpPointwiseConv, BpChannelExpansion, BpDepthwiseConv,
       BpHybridAtrouConv, BpSqueezeExcitation, BpMBConv
include("convolutions/convolutions.jl")

include("samplers.jl")

include("classifiers.jl")

#include("u_network.jl")

#export DropBlock
#include("dropblock.jl")

end
