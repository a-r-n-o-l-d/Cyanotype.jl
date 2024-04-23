#include("uchain.jl")

# Level parametric for ksize and connection path
@cyanotype begin
    """

    """
    struct UNet2Bp <: AbstractConvBp
        in_chs = 3
        nlvl = 4
        bwidth = 64
        expn = 2
        ksize = l -> 3
        enc
        dec
        bdg
        stem = nothing
        path = l -> nothing
        head = nothing
        top = nothing
        #connector = chcat
        residual = false
    end
end

function make(bp::UNet2Bp)
    # Build encoders and decoders for each level
    enc, dec, pth = [], [], []
    for l ∈ 1:bp.nlvl
        enclvl, declvl = _level_encodec(bp, bp.ksize(l), l)
        push!.((enc, dec), (enclvl, declvl))
        if !isnothing(bp.path(l))
            enc_chs, _ = _level_channels(bp, l)
            push!(pth, make(bp.path(l), bp.ksize(l), last(enc_chs)))
        else
            push!(pth, bp.path(l))
        end
    end
    bdg = make(bp.bdg, bp.ksize(bp.nlvl + 1), _bridge_channels(bp))
    return uchain(encoders=enc, decoders=dec, bridge=bdg, paths=pth)
end

function make(bp::UNet2Bp, ::Any, channels)
    tmp = spread(bp, in_chs=channels)
    make(tmp)
end
