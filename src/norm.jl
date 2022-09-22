abstract type AbstractCyanoNorm <: AbstractCyano end

@config struct CyanoIdentityNorm <: AbstractCyanoNorm end

build(channels, cfg::CyanoIdentityNorm) = Flux.identity


const bnmap = KwargsMapping(;
    field_names    = (:init_bias,   :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_names     = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types    = (:Function,    :Function,   :Bool,   :Bool,        :F,       :F),
    field_defaults = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0) )

@config bnmap struct CyanoBatchNorm{F <: CyanoFloat} <: AbstractCyanoNorm
    activation::Function = relu
end

