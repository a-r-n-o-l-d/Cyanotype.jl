include("uchain.jl")

# error with expn > 2: bridge channels are wrong

@cyanotype begin
    """

    """
    struct UEncoderBp
        conv
        downsampler
    end
end

make(bp::UEncoderBp, ksize, channels) = flatten_layers(
    [
        _make(bp.downsampler, channels), #_make(bp.downsampler, channels),
        make(bp.conv, ksize, channels)
    ]
)

@cyanotype begin
    """

    """
    struct UDecoderBp
        conv
        upsampler
    end
end

make(bp::UDecoderBp, ksize, channels) = flatten_layers(
    [
        make(bp.conv, ksize, channels),
        _make(bp.upsampler, channels)
    ]
)

@cyanotype begin
    """

    """
    struct UBridgeBp
        conv
        downsampler
        upsampler
    end
end

make(bp::UBridgeBp, ksize, channels) = flatten_layers(
    [
        _make(bp.downsampler, channels),
        make(bp.conv, ksize, channels),
        _make(bp.upsampler, channels)
    ]
)

@cyanotype begin
    """

    """
    struct UNetBp <: AbstractConvBp
        inchannels = 3
        nlevels = 4
        basewidth = 64
        expn = 2
        ksize = 3
        encoder
        decoder
        bridge
        stem = nothing # si nothing => encoder
        path = nothing #  connection_path (path CBAM, convpath)
        head = nothing # si nothing => decoder
        top = nothing
        #connector = chcat
        residual = false
    end
end

function make(bp::UNetBp)
    # Build encoders and decoders for each level
    if bp.residual
        enc, dec, pth = [], [], []
        for l ∈ 1:bp.nlevels
            enclvl, declvl = _level_encodec2(bp, bp.ksize, l)
            push!.((enc, dec), (enclvl, declvl))
            if !isnothing(bp.path)
                enc_chs, _ = _level_channels(bp, l)
                push!(pth, make(bp.path, bp.ksize, last(enc_chs)))
            else
                push!(pth, bp.path)
            end
        end
        bdg = make(bp.bridge, bp.ksize, _bridge_channels(bp))
        unet = uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
        enc_chs, dec_chs = _level_channels(bp, 1)
        stem = make(bp.stem, bp.ksize, bp.inchannels => first(enc_chs))
        head = make(bp.head, bp.ksize, last(dec_chs))
        top = make(bp.top, last(dec_chs))
        Chain(flatten_layers(stem)..., SkipConnection(unet, +), flatten_layers(head)..., flatten_layers(top)...)
    else
        enc, dec, pth = [], [], []
        for l ∈ 1:bp.nlevels
            enclvl, declvl = _level_encodec(bp, bp.ksize, l)
            push!.((enc, dec), (enclvl, declvl))
            if !isnothing(bp.path)
                enc_chs, _ = _level_channels(bp, l)
                push!(pth, make(bp.path, bp.ksize, last(enc_chs)))
            else
                push!(pth, bp.path)
            end
        end
        bdg = make(bp.bridge, bp.ksize, _bridge_channels(bp))
        uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
    end
end

function make(bp::UNetBp, ::Any, channels)
    tmp = spread(bp, inchannels=channels) #ksize=ksize,
    make(tmp)
end

############################################################################################
#                                   INTERNAL FUNCTIONS                                     #
############################################################################################

_make(bp, ksize, channels) = make(bp, ksize, channels)

_make(bp::ChannelAttentionBp, ksize, channels) = make(bp, channels)

_make(bp::PixelShuffleUpsamplerBp, channels) = make(bp, last(channels))

_make(bp::ConvTransposeUpsamplerBp, channels) = make(bp, last(channels) => last(channels) ÷ 2)

_make(bp, channels) = make(bp)

_make(bp::AbstractBpDownsampler, ::Any) = make(bp)

_make(bp::MeanMaxDownsamplerBp, ::Any) = make(bp, channels)

# Compute encoder/decoder number of channels at a given level (lvl)
# return two tuples one for encoder and one for decoder
# formula :
#  ice = expn^(lvl - 2) * basewidth
#  mce = expn^(lvl - 1) * basewidth
function _level_channels(bp, level)
    # encoder channels: input, middle, ouptput = (in_enc, mid_enc, out_enc)
    if level == 1
        if isnothing(bp.stem)
            in_enc = bp.inchannels
        else
            in_enc = bp.basewidth
        end
        mid_enc = out_enc = bp.basewidth
        in_dec = 2 * out_enc # channel concatenation
        mid_dec = out_dec = bp.basewidth
    else
        encp, decp = _level_channels(bp, level - 1)
        _, _, in_enc = encp
        mid_enc = out_enc = bp.expn * in_enc
        in_dec = 2 * out_enc
        mid_dec = in_dec ÷ bp.expn
        out_dec = bp.decoder.upsampler isa ConvTransposeUpsamplerBp ? mid_dec : mid_dec ÷ 2
    end
    (in_enc, mid_enc, out_enc), (in_dec, mid_dec, out_dec)
#=
    # encoder channels: input, middle, ouptput = (in_enc, mid_enc, out_enc)
    if level == 1
        if isnothing(bp.stem)
            in_enc = bp.inchannels
        else
            in_enc = bp.basewidth
        end
    else
        in_enc = bp.expn^(level - 2) * bp.basewidth
    end
    #in_enc = (level == 1) ? bp.inchannels : bp.expn^(level - 2) * bp.basewidth
    mid_enc = out_enc = bp.expn^(level - 1) * bp.basewidth
    # decoder channels: input, middle, ouptput = (icd, mcd, ocd)
    in_dec, mid_dec = 2 * out_enc, out_enc
    if bp.decoder.upsampler isa ConvTransposeUpsamplerBp
        out_dec = mid_dec
    else
        out_dec = (level == 1) ? mid_dec : mid_dec ÷ 2
    end
    (in_enc, mid_enc, out_enc), (in_dec, mid_dec, out_dec)
=#
end

function _bridge_channels(bp)
    #=
    enc, dec = _level_channels(bp, bp.nlevels + 1)
    in_chs, mid_chs, _ = enc
    _, _, out_chs = dec
    in_chs, mid_chs, out_chs
    =#
    enc, _ = _level_channels(bp, bp.nlevels + 1)
    _, dec = _level_channels(bp, bp.nlevels)
    in_chs, mid_chs, _ = enc
    out_chs, _, _ = dec
    #println(dec)
    out_chs = bp.decoder.upsampler isa ConvTransposeUpsamplerBp ? out_chs : out_chs ÷ 2
    in_chs, mid_chs, out_chs
end

function _level_encodec(bp, ksize, level) #
    # number of channels (input, middle, output)
    enc_chs, dec_chs = _level_channels(bp, level)
    if level == 1
        c = spread(bp.encoder.conv; stride=1)
        enc = [
                make(bp.stem, ksize, bp.inchannels => first(enc_chs)),
                make(c, ksize, enc_chs)
              ]
        #=
        if !isnothing(bp.stem)
            enc = [
                    make(bp.stem, ksize, bp.inchannels => last(enc_chs)),
                    make(bp.encoder.conv, ksize, enc_chs)
                  ]
        elseif isnothing(bp.encoder.downsampler) && isnothing(bp.stem)
            # if downsampling is done with a strided convolution
            e = spread(bp.encoder; stride=1)
            enc = make(e, ksize, enc_chs)
        else
            enc = make(bp.encoder.conv, ksize, enc_chs)
        end
        =#
        dec = [
                make(bp.decoder.conv, ksize, dec_chs),
                make(bp.head, ksize, last(dec_chs)),
                make(bp.top, last(dec_chs))
              ]
    else
        enc = make(bp.encoder, ksize, enc_chs)
        dec = make(bp.decoder, ksize, dec_chs)
    end
    flatten_layers(enc), flatten_layers(dec)
end

function _level_encodec2(bp, ksize, level)
    # number of channels (input, middle, output)
    enc_chs, dec_chs = _level_channels(bp, level)
    if level == 1
        c = spread(bp.encoder.conv; stride=1)
        enc = make(c, ksize, enc_chs)
        dec = make(bp.decoder.conv, ksize, dec_chs)
    else
        enc = make(bp.encoder, ksize, enc_chs)
        dec = make(bp.decoder, ksize, dec_chs)
    end
    flatten_layers(enc), flatten_layers(dec)
end
