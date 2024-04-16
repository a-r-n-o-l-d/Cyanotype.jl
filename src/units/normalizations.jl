abstract type AbstractNormBp <: AbstractBlueprint end

@cyanotype begin
    KwargsMapping(
            flfunc = :BatchNorm,
            fnames = (:initshift, :initscale, :affine, :trackstats,  :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :track_stats, :eps,     :momentum),
            defval = (zeros32,    ones32,     true,    true,         1f-5,     0.1f0)
        )

    """
    Wraps a Flux.Batchnorm
    """
    struct BatchNormBp <: AbstractNormBp
        @activation(identity)
    end
end

function make(bp::BatchNormBp, channels)
    [BatchNorm(channels, bp.activation; kwargs(bp)...)]
end

@cyanotype begin
    # !!!! `track_stats=true` will be removed from GroupNorm in Flux 0.14.
    KwargsMapping(
            flfunc = :GroupNorm,
            fnames = (:initshift, :initscale, :affine, :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :eps,     :momentum),
            defval = (zeros32,    ones32,      true,   1f-5,     0.1f0)
        )

    """
        GroupNormBp(; kwargs...)

    Describes a building process for a [`Groupnorm`](@ref Flux.Groupnorm) layer.
    make(channels, bp::CyGroupNorm)
    """
    struct GroupNormBp <: AbstractNormBp
        @activation(identity)
        """
        `groups`: the number of groups passed to [`GroupNorm`](@ref Flux.GroupNorm)
        constructor
        """
        groups
    end
end

function make(bp::GroupNormBp, channels)
    [GroupNorm(channels, bp.groups, bp.activation; kwargs(bp)...)]
end

@cyanotype begin
    KwargsMapping(
            flfunc = :InstanceNorm,
            fnames = (:initshift, :initscale, :affine, :trackstats,  :epsilon, :momentum),
            flargs = (:initβ,     :initγ,     :affine, :track_stats, :eps,     :momentum),
            defval = (zeros32,    ones32,     false,   false,        1f-5,     0.1f0)
        )

    """
    InstanceNormBp(; kwargs...)

    Describes a building process for a [`InstanceNorm`](@ref) layer.
    make(channels, bp::CyInstanceNorm)
    """
    struct InstanceNormBp <: AbstractNormBp
        @activation(identity)
    end
end

function make(bp::InstanceNormBp, channels)
    [InstanceNorm(channels, bp.activation; kwargs(bp)...)]
end
