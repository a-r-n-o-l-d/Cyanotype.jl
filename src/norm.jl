abstract type AbstractCyanoNorm <: AbstractCyano end

"""
    CyanoIdentityNorm()

Tagging `struct` indicating that no normalisation layer should be used in a building process.
"""
CyanoIdentityNorm

@cyano struct CyanoIdentityNorm end

"""
$(autogen_build(CyanoIdentityNorm, false, true))
"""
build(channels, cfg::CyanoIdentityNorm) = Flux.identity

#
const _NORMKW = (
field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)
)

register_mapping!(:bnmap=>KwargsMapping(; flux_function  = :BatchNorm, _NORMKW...,
additional_doc = "pouet pouet"
))

@cyano bnmap struct CyanoBatchNorm{F <: CyanoFloat} <: AbstractCyanoNorm
    """
    $ACTIVATION_DOC_RELU
    """
    activation = relu
end

# auto generation doc for build
function build(channels, cya::CyanoBatchNorm)
    kwargs = curate(cya)
    BatchNorm(channels, cya.activation; kwargs...)
end

register_mapping!(:gnmap=>KwargsMapping(; flux_function  = :GroupNorm, _NORMKW...))

@cyano gnmap struct CyanoGroupNorm{F <: CyanoFloat} <: AbstractCyanoNorm
    """
    $ACTIVATION_DOC_RELU
    """
    activation = relu
    """
    `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm) constructor
    """
    groups::Int
end

function build(channels, cya::CyanoGroupNorm)
    kwargs = curate(cya)
    GroupNorm(channels, cya.groups, cya.activation; kwargs...)
end
