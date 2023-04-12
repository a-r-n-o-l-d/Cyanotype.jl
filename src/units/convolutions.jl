abstract type AbstractBpConv <: AbstractBlueprint end

include("conv_units/conv.jl")
include("conv_units/compconv.jl")
include("conv_units/pwconv.jl")
include("conv_units/chconv.jl")
include("conv_units/dwconv.jl")
include("conv_units/haconv.jl")
include("conv_units/sqexc.jl")
include("conv_units/mbconv.jl")