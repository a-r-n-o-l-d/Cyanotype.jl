@cyanotype constructor=false begin
    """

    """
    struct PixelMapBp <: AbstractConvBp
        nmaps::Int
        projection::PointwiseConvBp
    end
end

PixelMapBp(; nmaps, kwargs...) = PixelMapBp(nmaps, PointwiseConvBp(; kwargs...))

make(bp::PixelMapBp, channels) = make(bp.projection, channels => bp.nmaps)
