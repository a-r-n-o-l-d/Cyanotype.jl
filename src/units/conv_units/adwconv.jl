@cyanotype begin #constructor=false
    """
    https://arxiv.org/pdf/2306.16103v2.pdf
    """
    struct AxialDWConvBp{A<:Function,
                         N<:Union{Nothing,AbstractNormBp},
                         P<:Union{SamePad,Int},
                         I<:Function} <: AbstractConvBp
        @volume
        @activation(identity)
        stride::Int = 1
        pad::P = SamePad()
        dilation::Int = 1
        init::I = glorot_uniform
        normalization::N = BatchNormBp()
    end
end
#=
AxialDWConvBp(; volume=false, activation=identity, stride=1, pad=SamePad(), dilation=1, init=glorot_uniform) = AxialDWConvBp(
    volume,
    activation,
    stride,
    pad,
    dilation,
    init
)
=#
make(bp::AxialDWConvBp, ksize, channels::Int) = make(bp, ksize, channels => channels)

function make(bp::AxialDWConvBp, ksize, channels::Pair)
    in_chs, out_chs = channels
    conv(k) = DepthwiseConv(k, in_chs => in_chs, stride=bp.stride, pad=bp.pad, dilation=bp.dilation, init=bp.init,
                            bias=bp.normalization isa Nothing)
    layers = []
    if bp.volume
        push!(layers, conv((ksize, 1, 1)))
        push!(layers, conv((1, ksize, 1)))
        push!(layers, conv((1, 1, ksize)))
    else
        push!(layers, conv((ksize, 1)))
        push!(layers, conv((1, ksize)))
    end
    norm = bp.normalization isa Nothing ? identity : BatchNorm(in_chs)
    pwc = PointwiseConvBp(activation=bp.activation, pad=bp.pad, init=bp.init)
    flatten_layers(
        [
            SkipConnection(Parallel(+, layers...), +),
            norm,
            make(pwc, in_chs => out_chs)
        ]
    )
end
