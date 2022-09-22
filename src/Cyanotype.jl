#=
wrappers: Flux to Cyanotype
modules: high level modules
=#
module Cyanotype

import Flux
using ExproniconLite
using Configurations
using Reexport

@reexport using Configurations: to_dict, from_dict

const CyanoFloat = Union{Float16, Float32, Float64}

include("utilities.jl")

#export register_mapping!
include("config.jl")

end
