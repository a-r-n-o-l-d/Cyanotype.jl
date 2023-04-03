#=
wrappers: Flux to Cyanotype
modules: high level modules
To do :
    - cleaning
    - rename fields with shorter names
    - replace NoNorm by nothing
    - ResidualConnection with dropout
=#
module Cyanotype

using Reexport
@reexport using Flux
#using CUDA
using Flux: zeros32, ones32, glorot_uniform
#using ChainRulesCore
#using Functors
#using MLUtils
#using Random

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

export BpConv, BpDConv, BpNConv, BpHAConv
include("convolutions.jl")

export BpSqueezeExcitation
include("squeeze_excitation.jl")

include("samplers.jl")

include("classifiers.jl")

include("u_network.jl")

#export DropBlock
#include("dropblock.jl")

end
