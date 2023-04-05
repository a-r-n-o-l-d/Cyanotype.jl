@cyanotype constructor=false begin
    """

    """
    struct BpChannelExpansion <: AbstractBpConv
        expansion::Int
        conv::BpPointwiseConv
    end
end

function BpChannelExpansion(; kwargs...)
    kw = Dict(kwargs)
    expansion = kw[:expansion]
    delete!(kw, :expansion)
    conv =  BpPointwiseConv(; kw...)
    BpChannelExpansion(expansion, conv)
end

function make(bp::BpChannelExpansion, channels)
    if bp.expansion <= 1
        identity
    else
        make(bp.conv, channels => channels * bp.expansion)
    end
end
