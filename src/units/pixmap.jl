@cyanotype constructor=false begin
    """

    """
    struct PixelMapBp <: AbstractConvBp
        nmaps
        proj
    end
end

PixelMapBp(; nmaps, kwargs...) = PixelMapBp(nmaps, PointwiseConvBp(; kwargs...))

make(bp::PixelMapBp, channels) = make(bp.proj, channels => bp.nmaps)
