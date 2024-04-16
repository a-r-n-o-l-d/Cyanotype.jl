@cyanotype constructor=false begin
    """

    """
    struct ChannelExpansionConvBp <: AbstractConvBp
        expansion#::Int
        conv#::PointwiseConvBp
    end
end

function ChannelExpansionConvBp(; kwargs...)
    kw = Dict(kwargs...)
    expansion = kw[:expansion]
    delete!(kw, :expansion)
    conv = PointwiseConvBp(; kw...)
    ChannelExpansionConvBp(expansion, conv)
end

function make(bp::ChannelExpansionConvBp, channels)
    if bp.expansion <= 1
        identity
    else
        make(bp.conv, channels => channels * bp.expansion)
    end
end
