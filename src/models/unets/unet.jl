include("uchain.jl")

@cyanotype begin
    """

    """
    struct UEncoderBp{C<:AbstractConvBp,D<:Union{Nothing,AbstractBpDownsampler}}
        convolution::C #DoubleConvBp
        downsampler::D #nothing si stride=2
    end
end

make(bp::UEncoderBp, ksize, channels) = flatten_layers(
    [
        make(bp.downsampler),
        make(bp.convolution, ksize, channels)
    ]
)


@cyanotype begin
    """

    """
    struct UDecoderBp{C<:AbstractConvBp,U<:AbstractBpUpsampler}
        convolution::C
        upsampler::U
    end
end

make(bp::UDecoderBp, ksize, channels) = flatten_layers(
    [
        make(bp.convolution, ksize, channels),
        _make(bp.upsampler, channels)
    ]
)

@cyanotype begin
    """

    """
    struct UBridgeBp{D<:Union{Nothing,AbstractBpDownsampler},U<:AbstractBpUpsampler} #,P<:BpPixelClassifierBp
        convolution::DoubleConvBp
        downsampler::D #nothing si stride=2
        upsampler::U
    end
end

make(bp::UBridgeBp, ksize, channels) = flatten_layers(
    [
        make(bp.downsampler),
        make(bp.convolution, ksize, channels),
        _make(bp.upsampler, channels)
    ]
)

@cyanotype begin
    """

    """
    struct UNetBp{S<:Union{Nothing,AbstractConvBp},
                  P<:Union{Nothing,AbstractConvBp},
                  H<:Union{Nothing,AbstractConvBp}}
        inchannels::Int = 3
        nlevels::Int = 4
        basewidth::Int = 64
        expansion::Int = 2
        ksize::Int = 3
        encoder::UEncoderBp
        decoder::UDecoderBp
        bridge::UBridgeBp
        stem::S = nothing # si nothing => encoder
        path::P = nothing #  => connection path CBAM, convpath
        head::H = nothing
    end
end

function make(bp::UNetBp)
    # Build encoders and decoders for each level
    enc, dec, pth = [], [], []
    for l ∈ 1:bp.nlevels
        enc_chs, dec_chs = _level_encodec(bp, 3, l)
        push!.((enc, dec), (enc_chs, dec_chs))
        if !isnothing(bp.path)
            push!(pth, make(bp.path, bp.ksize, last(enc_chs)))
        else
            push!(pth, bp.path)
        end
    end
    bdg = make(bp.bridge, 3, _bridge_channels(bp))
    uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

_make(bp::PixelShuffleUpsamplerBp, channels) = make(bp, last(channels))

_make(bp::ConvTransposeUpsamplerBp, channels) = make(bp, last(channels) => last(channels) ÷ 2)

_make(bp, channels) = make(bp)

# Compute encoder/decoder number of channels at a given level (lvl)
# return two tuples one for encoder and one for decoder
# formula :
#  ice = expansion^(lvl - 2) * basewidth
#  mce = expansion^(lvl - 1) * basewidth
function _level_channels(bp, level)
    # encoder channels: input, middle, ouptput = (in_enc, mid_enc, out_enc)
    in_enc = (level == 1) ? bp.inchannels : bp.expansion^(level - 2) * bp.basewidth
    mid_enc = out_enc = bp.expansion^(level - 1) * bp.basewidth
    # decoder channels: input, middle, ouptput = (icd, mcd, ocd)
    in_dec, mid_dec = 2 * out_enc, out_enc
    if bp.decoder.upsampler isa ConvTransposeUpsamplerBp
        out_dec = mid_dec
    else
        out_dec = (level == 1) ? mid_dec : mid_dec ÷ 2
    end
    (in_enc, mid_enc, out_enc), (in_dec, mid_dec, out_dec)
end

function _bridge_channels(bp)
    enc, dec = _level_channels(bp, bp.nlevels + 1)
    in_chs, mid_chs, _ = enc
    _, _, out_chs = dec
    in_chs, mid_chs, out_chs
end

function _level_encodec(bp, ksize, level) #
    # number of channels (input, middle, output)
    enc_chs, dec_chs = _level_channels(bp, level)
    if level == 1
        if !isnothing(bp.stem)
            enc = make(bp.stem, ksize, enc_chs)
        elseif isnothing(bp.encoder.downsampler) && isnothing(bp.stem)
            # if downsampling is done with a strided convolution
            encoder = spread(bp.encoder; stride=1)
            enc = make(encoder, ksize, enc_chs)
        else
            enc = make(bp.encoder.convolution, ksize, enc_chs)
        end
        dec = [
                make(bp.decoder.convolution, ksize, dec_chs),
                make(bp.head, last(dec_chs))
              ]
    else
        enc = make(bp.encoder, ksize, enc_chs)
        dec = make(bp.decoder, ksize, dec_chs)
    end
    flatten_layers(enc), flatten_layers(dec)
end
