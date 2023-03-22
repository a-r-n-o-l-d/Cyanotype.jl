const DEFUCONV = BpDConv(; conv1 = BpConv(;norm = BpBatchNorm()))

@cyanotype begin
    """

    """
    struct UModelBp{C<:BpDConv,D<:AbstractDownSamplerBp,U<:AbstractUpSamplerBp,
                    P<:PixelClassifierBp}
        @volumetric
        inchannels::Int
        nlevels::Int = 4
        basewidth::Int = 64
        expansion::Int = 2
        dconvolution::C = BpDConv(; vol = vol,
                                       conv1 = BpConv(; norm = BpBatchNorm())) #DEFUCONV pad = 0
        downsampler::D = MaxDownSamplerBp(; vol = vol)
        upsampler::U = LinearUpSamplerBp(; vol = vol)
        classifier::P
        connection = chcat
    end
end

function make(bp::UModelBp)
    # verifier vol
    # Build encoders and decoders for each level
    encoders, decoders, connections = [], [], []
    for l ∈ 1:bp.nlevels
        push!.((encoders, decoders), _level_encodec(bp, l))
        push!(connections, chcat)
        #=
        if padding
            push!(con, chcat)
        else
            push!(con, CenterCropCat(trim(l, nlevels), volume))
        end
        =#
    end

    # Build bridge (i.e. bottom part)
    enc_chs, dec_chs = _level_channels(bp, nlevels + 1)
    in_chs, mid_chs, _ = enc_chs
    _, _, out_chs = dec_chs
    #dw, up = samplers(out_chs, pars)
    bridge  = [
                make(bp.downsampler),
                make(bp.dconvolution; channels = (in_chs, mid_chs, out_chs)),
                make(bp.upsample; channels = out_chs=>out_chs ÷ 2) # ne fonctionne que pour convtranspose, out_chs=>out_chs ÷ 2 uniquement concatenation, si addition out_chs=>out_chs
              ]
end

function _upsampler(bp, channels)
    if bp.upsampler isa ConvUpSamplerBp
        if bp.connections === chcat || CenterCropCat
            make(bp.upsample; channels = out_chs=>out_chs ÷ 2)
        else
            make(bp.upsample; channels = out_chs=>out_chs)
        end
    else
        make(bp.upsample)
    end
end

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
    if bp.downsampler isa ConvUpSamplerBp
        out_dec = mid_dec
    else
        out_dec = (level == 1) ? mid_dec : mid_dec ÷ 2
    end
    (in_enc, mid_enc, out_enc), (in_dec, mid_dec, out_dec)
end

function _level_encodec(bp, level)
    # number of channels (input, middle, output)
    enc_chs, dec_chs = _level_channels(bp, level)
    if level == 1
        enc = make(bp.dconvolution; channels = enc_chs)
        dec = [
                make(bp.dconvolution; channels = dec_chs),
                make(bp.classifier; channels = dec_chs[3])
              ]
    else
        enc = [
                make(bp.downsampler),
                make(bp.dconvolution; channels = enc_chs)
              ]
        dec = [
                make(bp.dconvolution; channels = dec_chs),
                make(bp.upsampler; channels = dec_chs[3]=>dec_chs[3] ÷ 2) # ne fonctionne que pour convtranspose, out_chs=>out_chs ÷ 2 uniquement concatenation, si addition out_chs=>out_chs
              ]
    end
    flatten_layers(enc), flatten_layers(dec)
end
