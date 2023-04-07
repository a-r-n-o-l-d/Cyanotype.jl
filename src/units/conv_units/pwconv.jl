@cyanotype constructor=false begin
    """

    """
    struct BpPointwiseConv <: AbstractBpConv
        conv::BpConv
    end
end

BpPointwiseConv(; kwargs...) = BpPointwiseConv(BpConv(; kwargs...))

make(bp::BpPointwiseConv, channels) = make(bp.conv, 1, channels)
