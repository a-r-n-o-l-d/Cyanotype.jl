@cyanotype constructor=false begin
    """

    """
    struct BpPointwiseConv <: AbstractConvBp
        conv::ConvBp
    end
end

BpPointwiseConv(; kwargs...) = BpPointwiseConv(ConvBp(; kwargs...))

make(bp::BpPointwiseConv, channels) = make(bp.conv, 1, channels)
