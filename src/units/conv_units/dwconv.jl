@cyanotype constructor=false begin
    """

    """
    struct BpDepthwiseConv <: AbstractConvBp
        conv::ConvBp
    end
end

function BpDepthwiseConv(; kwargs...)
    kw = Dict(kwargs...)
    #if haskey(kw, :depthwise)
        #kw[:depthwise] = true
    #end
    BpDepthwiseConv(ConvBp(; depthwise=true, kw...))
end

make(bp::BpDepthwiseConv, ksize, channels) = make(bp.conv, ksize, channels)
