@cyanotype begin
    """

    """
    struct EfficientNetStageBp{C<:AbstractConvBp,
                               R1<:Union{Nothing,Real},
                               R2<:Union{Nothing,Real}} <: AbstractConvBp
        ksize::Int
        outchannels::Int
        nrepeat::Int
        convolution::C # BpMBConv(; stride, ch_expansion, se_reduction, activation)
        widthscaling::R1 = nothing
        depthscaling::R2 = nothing
    end
end

EfficientNetStageBp(::Type{FusedMbConvBp}, ksize, out_chs, expansion, stride,
                    nrepeat) = EfficientNetStageBp(
    ksize=ksize,
    outchannels=out_chs,
    nrepeat=nrepeat,
    convolution=FusedMbConvBp(
        stride=stride,
        ch_expansion=expansion,
        activation=swish
    )
)


EfficientNetStageBp(::Type{MbConvBp}, ksize, out_chs, expansion, stride, nrepeat,
                    reduction, wscaling=nothing, dscaling=nothing) = EfficientNetStageBp(
    ksize=ksize,
    outchannels=_out_channels(wscaling, out_chs),
    nrepeat=_nrepeats(dscaling, nrepeat), #isnothing(dscaling) ? nrepeat : ceil(Int, nrepeat * dscaling)
    widthscaling=wscaling,
    depthscaling=dscaling,
    convolution=MbConvBp(
        stride=stride,
        ch_expansion=expansion,
        se_reduction=reduction,
        activation=swish
    )
)

function make(bp::EfficientNetStageBp, channels::Int)
    in_chs = _round_channels(channels)
    #out_chs = isnothing(bp.widthscaling) ? _round_channels(bp.outchannels) : _round_channels(bp.outchannels * bp.widthscaling)
    layers = []
    push!(layers, make(bp.convolution, bp.ksize, in_chs => bp.outchannels))
    #nrepeat = isnothing(bp.depthscaling) ? bp.nrepeat - 1 : ceil(Int, bp.nrepeat * bp.depthscaling) - 1
    for _ in 1:bp.nrepeat
        conv = spread(bp.convolution; stride = 1, skip=true)
        push!(layers, make(conv, bp.ksize, bp.outchannels => bp.outchannels))
    end
    flatten_layers(layers)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

_nrepeats(::Type{Nothing}, n) = n - 1

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
