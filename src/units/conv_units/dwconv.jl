@cyanotype constructor=false begin
    """

    """
    struct DepthwiseConvBp <: AbstractConvBp
        conv
    end
end

function DepthwiseConvBp(; kwargs...)
    kw = Dict(kwargs...)
    #if haskey(kw, :dwise)
        #kw[:dwise] = true
    #end
    DepthwiseConvBp(ConvBp(; dwise=true, kw...))
end

make(bp::DepthwiseConvBp, ksize, channels) = make(bp.conv, ksize, channels)
