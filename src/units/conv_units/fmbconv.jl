@cyanotype constructor=false begin
    """

    """
    struct FusedMbConvBp <: AbstractConvBp
        skip
        expn_ch
        conv
        #dropout
        proj
    end
end

FusedMbConvBp(; stride, expn_ch, skip=(stride == 1), act=relu, norm=BatchNormBp(act=act),
              kwargs...) = FusedMbConvBp(
    skip,
    expn_ch,
    ConvBp(; stride=stride, act=act, norm=norm, kwargs...),
    expn_ch <= 1 ? nothing : PointwiseConvBp(; norm=norm, kwargs...)
)

function make(bp::FusedMbConvBp, ksize, channels, dropout=0)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.expn_ch
    layers = flatten_layers(
        [
            make(bp.conv, ksize, in_chs => mid_chs),
            make(bp.proj, mid_chs => out_chs)
        ]
    )
    if bp.skip && in_chs == out_chs
        if iszero(dropout)
            SkipConnection(Chain(layers...), +)
        else
            d = bp.proj.conv.vol ? 5 : 4
            Parallel(+, Chain(layers...), Dropout(dropout, dims=d))
        end
    else
        layers
    end
end
