@cyanotype constructor=false begin
    """

    """
    struct ChannelExpansionConvBp <: AbstractConvBp
        expn
        conv
    end
end

function ChannelExpansionConvBp(; kwargs...)
    kw = Dict(kwargs...)
    expn = kw[:expn]
    delete!(kw, :expn)
    conv = PointwiseConvBp(; kw...)
    ChannelExpansionConvBp(expn, conv)
end

function make(bp::ChannelExpansionConvBp, channels)
    if bp.expn <= 1
        identity
    else
        make(bp.conv, channels => channels * bp.expn)
    end
end
