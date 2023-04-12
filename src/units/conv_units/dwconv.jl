@cyanotype constructor=false begin
    """

    """
    struct BpDepthwiseConv <: AbstractBpConv
        conv::BpConv
    end
end

function BpDepthwiseConv(; kwargs...)
    kw = Dict(kwargs)
    #if haskey(kw, :depthwise)
        kw[:depthwise] = true
    #end
    BpDepthwiseConv(BpConv(; kw...))
end

make(bp::BpDepthwiseConv, ksize, channels) = make(bp.conv, ksize, channels)
