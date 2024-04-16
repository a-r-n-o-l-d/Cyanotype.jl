@cyanotype begin
    """
    https://arxiv.org/pdf/2306.16103v2.pdf
    """
    struct AxialDWConvBp <: AbstractConvBp
        @volume
        @activation(identity)
        stride        = 1
        pad           = SamePad()
        dilation      = 1
        init          = glorot_uniform
        norm = BatchNormBp()
        skip          = true
    end
end

make(bp::AxialDWConvBp, ksize, channels::Int) = make(bp, ksize, channels => channels)

function make(bp::AxialDWConvBp, ksize, channels::Pair)
    in_chs, out_chs = channels
    conv(k) = DepthwiseConv(k, in_chs => in_chs, stride=bp.stride, pad=bp.pad, dilation=bp.dilation, init=bp.init,
                            bias=bp.norm isa Nothing)
    layers = []
    if bp.vol
        push!(layers, conv((ksize, 1, 1)))
        push!(layers, conv((1, ksize, 1)))
        push!(layers, conv((1, 1, ksize)))
    else
        push!(layers, conv((ksize, 1)))
        push!(layers, conv((1, ksize)))
    end
    norm = bp.norm isa Nothing ? identity : BatchNorm(in_chs)
    pwc = PointwiseConvBp(act=bp.act, pad=bp.pad, init=bp.init)
    axial = bp.skip ? SkipConnection(Parallel(+, layers...), +) : Parallel(+, layers...)
    flatten_layers(
        [
            axial,
            norm,
            make(pwc, in_chs => out_chs)
        ]
    )
end
