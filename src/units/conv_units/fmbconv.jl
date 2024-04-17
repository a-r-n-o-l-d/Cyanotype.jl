@cyanotype constructor=false begin
    """

    """
    struct FusedMbConvBp <: AbstractConvBp
        skip
        ch_expansion
        conv
        #dropout
        projection
    end
end

FusedMbConvBp(; stride, ch_expansion, skip=(stride == 1), act=relu,
norm=BatchNormBp(act=act),
                         kwargs...) = FusedMbConvBp(
    skip,
    ch_expansion,
    ConvBp(; stride=stride, act=act, norm=norm, kwargs...),
    ch_expansion <= 1 ? nothing : PointwiseConvBp(; norm=norm, kwargs...)
)

function make(bp::FusedMbConvBp, ksize, channels, dropout=0)
    in_chs, out_chs = channels
    mid_chs = in_chs * bp.ch_expansion
    #=if bp.skip && in_chs !== out_chs
        error("""
        If a 'FusedMbConvBp' have a skip connection defined, the number fo input channels and
        output channels must be the same.
        """)
    end=#
    layers = flatten_layers(
        [
            make(bp.conv, ksize, in_chs => mid_chs),
            make(bp.projection, mid_chs => out_chs)
        ]
    )
    #bp.skip && in_chs == out_chs ? SkipConnection(Chain(layers...), +) : layers
    if bp.skip && in_chs == out_chs
        if iszero(dropout)
            SkipConnection(Chain(layers...), +)
        else
            d = bp.projection.conv.vol ? 5 : 4
            Parallel(+, Chain(layers...), Dropout(dropout, dims=d))
        end
    else
        layers
    end
end
