@cyanotype constructor=false begin
    """

    """
    struct BpChannelExpansionConv <: AbstractConvBp
        expansion::Int
        conv::PointwiseConvBp
    end
end

function BpChannelExpansionConv(; kwargs...)
    kw = Dict(kwargs...)
    expansion = kw[:expansion]
    delete!(kw, :expansion)
    conv =  PointwiseConvBp(; kw...)
    BpChannelExpansionConv(expansion, conv)
end

function make(bp::BpChannelExpansionConv, channels)
    if bp.expansion <= 1
        identity
    else
        make(bp.conv, channels => channels * bp.expansion)
    end
end
