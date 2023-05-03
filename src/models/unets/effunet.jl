@cyanotype begin
    """

    """
    struct EfficientUNetBp{H<:Union{Nothing,AbstractConvBp}}
        backbone::EfficientNetBp
        decoder::UDecoderBp
        head::H = nothing
    end
end

function make(bp::EfficientUNetBp)
    effnet = bp.backbone

    bdg_chs = effnet.backbone[end].outchannels
    get_stride(conv::MbConvBp) = conv.depthwise.conv.stride
    get_stride(conv::FusedMbConvBp) = conv.convolution.stride
    nlevel = 1
    for s in effnet.backbone
        if get_stride(s.convolution) == 2
            nlevel = nlevel + 1
        end
    end
    nlevel = nlevel - 1
    dec_in_chs(level) = out_chs + dec_out_chs(level + 1) #
    dec_out_chs(level) = level == nlevel + 1 ? bdg_chs : 2^(level + 4)
    out_chs = effnet.stemchannels
    stem = make(effnet.stem, 3, effnet.inchannels => out_chs)
    encoders = [Any[stem]]
    decoders = []
    level = 1
    for s in effnet.backbone
        stage = make(s, out_chs)
        if get_stride(s.convolution) == 1
            push!(last(encoders), stage)
        else
            push!(encoders, Any[stage])
            dec = make(bp.decoder, 3, dec_in_chs(level) => dec_out_chs(level))
            if level == 1
                push!(dec, make(bp.head, dec_out_chs(level)))
            end
            push!(decoders, dec)
            level = level + 1
        end
        out_chs = s.outchannels
    end
    bridge = [pop!(encoders), make(PixelShuffleUpsamplerBp(), out_chs)]
    uchain(
        encoders=map(flatten_layers, encoders),
        decoders=map(flatten_layers, decoders),
        bridge=flatten_layers(bridge)
    )
end
