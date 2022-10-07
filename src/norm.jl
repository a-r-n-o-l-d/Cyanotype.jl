abstract type AbstractNormBp <: AbstractBlueprint end

@cyanotype (
"""
    NoNormBp()
"""
) (
struct NoNormBp{A} <: AbstractNormBp
    @activation(Flux.relu)
end
)

@cyanotype (
"""
Wraps a Flux.Batchnorm
"""
) (
KwargsMapping(; flux_function = :BatchNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, true,    true,         1f-5,     0.1f0))
) (
struct BatchNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
    @activation(relu)
end
)

function make(bp::BatchNormBp, channels)
    BatchNorm(channels, bp.activation; kwargs(bp)...)
end

make(bp::BatchNormBp; channels) = make(bp, channels)

@cyanotype (
"""
    GroupNormBp(; kwargs...)

Describes a building process for a [`Groupnorm`](@ref Flux.Groupnorm) layer.
make(channels, bp::CyGroupNorm)
"""
) (
# !!!! `track_stats=true` will be removed from GroupNorm in Flux 0.14.
KwargsMapping(;
    flux_function = :GroupNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, true,    false,        1f-5,     0.1f0))
) (
struct GroupNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
    @activation(relu)
    """
    `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm) constructor
    """
    groups::Int
end
)

function make(bp::GroupNormBp, channels)
    GroupNorm(channels, bp.groups, bp.activation; kwargs(bp)...)
end

make(bp::GroupNormBp; channels) = make(bp, channels)

@cyanotype (
"""
InstanceNormBp(; kwargs...)

Describes a building process for a [`InstanceNorm`](@ref Flux.InstanceNorm) layer.
make(channels, bp::CyInstanceNorm)
"""
) (
KwargsMapping(;
    flux_function = :InstanceNorm,
    field_names = (:init_shift,  :init_scale, :affine, :track_stats, :epsilon, :momentum),
    flux_kwargs = (:initβ,       :initγ,      :affine, :track_stats, :ϵ,       :momentum),
    field_types = (:I1,          :I2,         Bool,    Bool,         :F,       :F),
    def_values  = (Flux.zeros32, Flux.ones32, false,    false,        1f-5,     0.1f0))
) (
struct InstanceNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
    @activation(relu)
end
)

function make(bp::InstanceNormBp, channels)
    InstanceNorm(channels, bp.activation; kwargs(bp)...)
end

make(bp::InstanceNormBp; channels) = make(bp, channels)
