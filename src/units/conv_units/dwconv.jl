@cyanotype constructor=false begin
    """

    """
    struct DepthwiseConvBp <: AbstractConvBp
        conv
    end
end

function DepthwiseConvBp(; kwargs...)
    kw = Dict(kwargs...)
    #if haskey(kw, :dw)
        #kw[:dw] = true
    #end
    DepthwiseConvBp(ConvBp(; dw=true, kw...))
end

make(bp::DepthwiseConvBp, ksize, channels) = make(bp.conv, ksize, channels)
