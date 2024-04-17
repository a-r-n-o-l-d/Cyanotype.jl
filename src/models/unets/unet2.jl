#include("uchain.jl")

# Level parametric for ksize and connection path
@cyanotype begin
    """

    """
    struct UNet2Bp <: AbstractConvBp
        inchannels = 3
        nlevels = 4
        basewidth = 64
        expansion = 2
        ksize = l -> 3
        encoder
        decoder
        bridge
        stem = nothing # si nothing => encoder
        path = l -> nothing #  connection_path (path CBAM, convpath)
        head = nothing # si nothing => decoder
        top = nothing
        #connector = chcat
        residual = false
    end
end

function make(bp::UNet2Bp)
    # Build encoders and decoders for each level
    if bp.residual
        enc, dec, pth = [], [], []
        for l ∈ 1:bp.nlevels
            enclvl, declvl = _level_encodec2(bp, bp.ksize(l), l)
            push!.((enc, dec), (enclvl, declvl))
            if !isnothing(bp.path(l))
                enc_chs, _ = _level_channels(bp, l)
                push!(pth, make(bp.path(l), bp.ksize(l), last(enc_chs)))
            else
                push!(pth, bp.path(l))
            end
        end
        bdg = make(bp.bridge, bp.ksize(bp.nlevels + 1), _bridge_channels(bp))
        unet = uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
        enc_chs, dec_chs = _level_channels(bp, 1)
        stem = make(bp.stem, bp.ksize(1), bp.inchannels => first(enc_chs))
        head = make(bp.head, bp.ksize(1), last(dec_chs))
        top = make(bp.top, last(dec_chs))
        Chain(flatten_layers(stem)..., SkipConnection(unet, +), flatten_layers(head)..., flatten_layers(top)...)
    else
        enc, dec, pth = [], [], []
        for l ∈ 1:bp.nlevels
            enclvl, declvl = _level_encodec(bp, bp.ksize(l), l)
            push!.((enc, dec), (enclvl, declvl))
            if !isnothing(bp.path(l))
                enc_chs, _ = _level_channels(bp, l)
                push!(pth, make(bp.path(l), bp.ksize(l), last(enc_chs)))
            else
                push!(pth, bp.path(l))
            end
        end
        bdg = make(bp.bridge, bp.ksize(bp.nlevels + 1), _bridge_channels(bp))
        unet = uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
        unet
    end
end

function make(bp::UNet2Bp, ::Any, channels)
    tmp = spread(bp, inchannels=channels) #ksize=ksize,
    make(tmp)
end