@cyanotype constructor=false begin
    """

    """
    struct PointwiseConvBp{C<:AbstractConvBp} <: AbstractConvBp
        conv::C
    end
end

PointwiseConvBp(; kwargs...) = PointwiseConvBp(ConvBp(; kwargs...))

make(bp::PointwiseConvBp, channels) = make(bp.conv, 1, channels)

make(bp::PointwiseConvBp, ksize, channels) = make(bp, channels)
