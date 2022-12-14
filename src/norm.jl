abstract type AbstractNormBp <: AbstractBlueprint end

@cyanotype begin
    """
        NoNormBp()
    """
    struct NoNormBp{A} <: AbstractNormBp
        @activation(Flux.relu)
    end
end

@cyanotype begin
    KwargsMapping(; flux_function = :BatchNorm,
        field_names = (:init_shift, :init_scale, :affine, :track_stats, :epsilon, :momentum),
        flux_kwargs = (:initβ,      :initγ,      :affine, :track_stats, :ϵ,       :momentum),
        field_types = (:I1,         :I2,         Bool,    Bool,         :F,       :F),
        def_values  = (zeros32,     ones32,      true,    true,         1f-5,     0.1f0))

    """
    Wraps a Flux.Batchnorm
    """
    struct BatchNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
        @activation(relu)
    end
end

function make(bp::BatchNormBp; channels)
    BatchNorm(channels, bp.activation; kwargs(bp)...)
end

@cyanotype begin
    # !!!! `track_stats=true` will be removed from GroupNorm in Flux 0.14.
    KwargsMapping(;
        flux_function = :GroupNorm,
        field_names = (:init_shift, :init_scale, :affine, :track_stats, :epsilon, :momentum),
        flux_kwargs = (:initβ,      :initγ,      :affine, :track_stats, :ϵ,       :momentum),
        field_types = (:I1,         :I2,         Bool,    Bool,         :F,       :F),
        def_values  = (zeros32,     ones32,      true,    false,        1f-5,     0.1f0))

    """
        GroupNormBp(; kwargs...)

    Describes a building process for a [`Groupnorm`](@ref Flux.Groupnorm) layer.
    make(channels, bp::CyGroupNorm)
    """
    struct GroupNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
        @activation(relu)
        """
        `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm)
        constructor
        """
        groups::Int
    end
end

function make(bp::GroupNormBp; channels)
    GroupNorm(channels, bp.groups, bp.activation; kwargs(bp)...)
end

@cyanotype begin
    KwargsMapping(;
        flux_function = :InstanceNorm,
        field_names = (:init_shift, :init_scale, :affine, :track_stats, :epsilon, :momentum),
        flux_kwargs = (:initβ,      :initγ,      :affine, :track_stats, :ϵ,       :momentum),
        field_types = (:I1,         :I2,         Bool,    Bool,         :F,       :F),
        def_values  = (zeros32,     ones32,      false,   false,        1f-5,     0.1f0))

    """
    InstanceNormBp(; kwargs...)

    Describes a building process for a [`InstanceNorm`](@ref) layer.
    make(channels, bp::CyInstanceNorm)
    """
    struct InstanceNormBp{F<:CyFloat,A<:Function,I1<:Function,I2<:Function} <: AbstractNormBp
        @activation(relu)
    end
end

function make(bp::InstanceNormBp; channels)
    InstanceNorm(channels, bp.activation; kwargs(bp)...)
end
