@cyanotype begin
    """

    """
    struct EfficientNetStageBp <: AbstractConvBp
        ksize
        out_chs
        nrepeat
        conv
        wscaling = nothing
        dscaling = nothing
    end
end

EfficientNetStageBp(::Type{FusedMbConvBp}, ksize, out_chs, expn, stride,
                    nrepeat) = EfficientNetStageBp(
    ksize=ksize,
    out_chs=out_chs,
    nrepeat=_nrepeats(nrepeat),
    conv=FusedMbConvBp(
        stride=stride,
        ch_expn=expn,
        act=swish
    )
)

EfficientNetStageBp(::Type{MbConvBp}, ksize, out_chs, expn, stride, nrepeat,
                    reduc, wscaling=nothing, dscaling=nothing) = EfficientNetStageBp(
    ksize=ksize,
    out_chs=_out_channels(wscaling, out_chs),
    nrepeat=_nrepeats(dscaling, nrepeat),
    wscaling=wscaling,
    dscaling=dscaling,
    conv=MbConvBp(
        stride=stride,
        ch_expn=expn,
        se_reduction=reduc,
        act=swish
    )
)

function make(bp::EfficientNetStageBp, channels::Int, dropouts=zeros(bp.nrepeat + 1))
    in_chs = _round_channels(channels)
    layers = []
    push!(layers, make(bp.conv, bp.ksize, in_chs => bp.out_chs, dropouts[1]))
    for d in dropouts[2:end] #_ in 1:bp.nrepeat
        conv = spread(bp.conv; stride = 1, skip=true)
        push!(layers, make(conv, bp.ksize, bp.out_chs => bp.out_chs, d))
    end
    flatten_layers(layers)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

_nrepeats(n) = n - 1

_nrepeats(::Nothing, n) = n - 1

_nrepeats(scaling, n) = ceil(Int, n * scaling) - 1

function _out_channels(scaling, channels)
    if isnothing(scaling)
        _round_channels(channels)
    else
        _round_channels(channels * scaling)
    end
end

# From Metalhead.jl
# utility function for making sure that all layers have a channel size divisible by 8
# used by MobileNet variants
function _round_channels(channels::Number, divisor::Integer=8, min_value::Integer=0)
    new_channels = max(min_value, floor(Int, channels + divisor / 2) ÷ divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    return new_channels < 0.9 * channels ? new_channels + divisor : new_channels
end
