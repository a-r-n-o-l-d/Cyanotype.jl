#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

using Flux
using Markdown: MD

export build, CyFloat

const CyFloat = Union{Float16, Float32, Float64}

include("utilities.jl")

include("cyanotype.jl")

#include("kwmapping.jl")
#include("cyano.jl")
#export register_mapping!
#include("config.jl")

export CyIdentityNorm, CyBatchNorm, CyGroupNorm
include("norm.jl")

export CyConv
include("conv.jl")

end
