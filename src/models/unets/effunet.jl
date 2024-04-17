@cyanotype begin
    """

    """
    struct EfficientUNetBp
        bbone
        dec
        head = nothing
    end
end

function make(bp::EfficientUNetBp)
    effnet = bp.bbone
    bdg_chs = effnet.bbone[end].out_chs
    get_stride(conv::MbConvBp) = conv.dwise.conv.stride
    get_stride(conv::FusedMbConvBp) = conv.conv.stride
    nlevel = 1
    for s in effnet.bbone
        if get_stride(s.conv) == 2
            nlevel = nlevel + 1
        end
    end
    nlevel = nlevel - 1
    dec_in_chs(level) = out_chs + dec_out_chs(level + 1) #
    dec_out_chs(level) = level == nlevel + 1 ? bdg_chs : 2^(level + 4)
    out_chs = effnet.st_chs
    stem = make(effnet.stem, 3, effnet.in_chs => out_chs)
    encoders = [Any[stem]]
    decoders = []
    level = 1
    for s in effnet.bbone
        stage = make(s, out_chs)
        if get_stride(s.conv) == 1
            push!(last(encoders), stage)
        else
            push!(encoders, Any[stage])
            dec = make(bp.dec, 3, dec_in_chs(level) => dec_out_chs(level))
            if level == 1
                push!(dec, make(bp.head, dec_out_chs(level)))
            end
            push!(decoders, dec)
            level = level + 1
        end
        out_chs = s.out_chs
    end
    bridge = [pop!(encoders), make(PixelShuffleUpsamplerBp(), out_chs)]
    uchain(
        encoders=map(flatten_layers, encoders),
        decoders=map(flatten_layers, decoders),
        bridge=flatten_layers(bridge)
    )
end
