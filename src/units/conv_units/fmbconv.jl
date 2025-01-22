@cyanotype constructor=false begin
    """

    """
    struct FusedMbConvBp <: AbstractConvBp
        skip
        exch
        conv
        proj
    end
end

FusedMbConvBp(; stride, exch, skip=(stride == 1), act=relu, norm=BatchNormBp(act=act),
              kwargs...) = FusedMbConvBp(
    skip,
    exch,
    ConvBp(; stride=stride, act=act, norm=norm, kwargs...),
    exch <= 1 ? nothing : PointwiseConvBp(; norm=norm, kwargs...)
)

function make(bp::FusedMbConvBp, ksize, channels; dropout=0)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.exch
    layers = flatten_layers(
        [
            make(bp.conv, ksize, in_chs => mid_chs),
            make(bp.proj, mid_chs => out_chs)
        ]
    )
    if bp.skip && in_chs == out_chs
        if iszero(dropout)
            return SkipConnection(Chain(layers...), +)
        else
            d = bp.conv.vol ? 5 : 4
            return Parallel(+, Chain(layers...), Dropout(dropout, dims=d))
        end
    else
        return layers
    end
end
