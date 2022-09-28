abstract type AbstractCyanoNorm <: AbstractCyano end

@cyano struct CyanoIdentityNorm  end #<: AbstractCyanoNorm

build(channels, cfg::CyanoIdentityNorm) = Flux.identity

#=const BNMAP = KwargsMapping(;
flux_function  = :BatchNorm,
field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0))
register_mapping!(:BNMAP=>BNMAP)=#

register_mapping!(:bnmap=>KwargsMapping(;
flux_function  = :BatchNorm,
field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0)))

@cyano bnmap struct CyanoBatchNorm{F <: CyanoFloat} <: AbstractCyanoNorm
    """ Activation function, by default ['relu'](@ref Flux.relu) """
    activation::Function = relu
end

# auto generation doc for build
function build(channels, cya::CyanoBatchNorm)
    kwargs = curate(cya)
    BatchNorm(channels, cya.activation; kwargs...)
end

#=const GNMAP = KwargsMapping(;
flux_function  = :GroupNorm,
field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
field_defaults = (Flux.zeros32, Flux.ones32, true,    false,        1f-5,     0.1f0))
register_mapping!(:GNMAP=>GNMAP)=#
register_mapping!(:gnmap=>KwargsMapping(;
flux_function  = :GroupNorm,
field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
field_defaults = (Flux.zeros32, Flux.ones32, true,    false,        1f-5,     0.1f0)))

@cyano gnmap struct CyanoGroupNorm{F <: CyanoFloat} <: AbstractCyanoNorm
    activation::Function = relu
    """ groups is the number of groups """
    groups::Int
end

function build(channels, cya::CyanoGroupNorm)
    kwargs = curate(cya)
    GroupNorm(channels, cya.groups, cya.activation; kwargs...)
end
