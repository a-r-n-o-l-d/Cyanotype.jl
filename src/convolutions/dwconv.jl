@cyanotype constructor=false begin
    """

    """
    struct BpDepthwiseConv <: AbstractBpConv
        conv::BpConv
    end
end

function BpDepthwiseConv(; kwargs...)
    if haskey(kwargs, :depthwise)
        kwargs[:depthwise] = true
    end
    BpDepthwiseConv(BpConv(; kwargs...))
end

make(bp::BpDepthwiseConv, ksize, channels) = make(bp.conv, ksize, channels)
