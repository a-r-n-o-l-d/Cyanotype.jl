abstract type AbstractCyNorm <: AbstractCyanotype end

"""
    CyanoIdentityNorm()

Tagging `struct` indicating that no normalisation layer should be used in a building process.
"""
CyIdentityNorm

@cyano struct CyIdentityNorm <: AbstractCyNorm end

"""
$(autogen_build_doc(CyIdentityNorm, false, true))
"""
build(::Any, ::CyIdentityNorm) = Flux.identity #inutile

# Defines kwargs for Flux normalisation layers
const _NORMKW = (
field_names    = (:init_shift,   :init_scale, :affine, :track_stats, :epsilon, :momentum), #init_bias => init_shift
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Any,         :Any,        :Bool,   :Bool,        :F,       :F), # Function => type instbility, wrap in CyFunc => curate doit etre modifee
field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)
)

register_mapping!(:bnmap=>KwargsMapping(; flux_function  = :BatchNorm, _NORMKW...))

@cyano bnmap struct CyBatchNorm{F <: CyanoFloat} <: AbstractCyNorm
    @activation(relu)
end

# auto generation doc for build
function build(channels, cya::CyBatchNorm)
    kwargs = curate(cya)
    BatchNorm(channels, cya.activation; kwargs...)
end

register_mapping!(:gnmap=>KwargsMapping(; flux_function  = :GroupNorm, _NORMKW...))

@cyano gnmap struct CyGroupNorm{F <: CyanoFloat} <: AbstractCyNorm
    @activation(relu)

    """
    `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm) constructor
    """
    groups::Int
end

function build(channels, cya::CyGroupNorm)
    kwargs = curate(cya)
    GroupNorm(channels, cya.groups, cya.activation; kwargs...)
end
