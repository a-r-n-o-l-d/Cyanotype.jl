@cyanotype constructor=false begin
    """

    """
    struct DepthwiseConvBp <: AbstractConvBp
        conv
    end
end

function DepthwiseConvBp(; kwargs...)
    kw = Dict(kwargs...)
    #if haskey(kw, :depthwise)
        #kw[:depthwise] = true
    #end
    DepthwiseConvBp(ConvBp(; depthwise=true, kw...))
end

make(bp::DepthwiseConvBp, ksize, channels) = make(bp.conv, ksize, channels)
