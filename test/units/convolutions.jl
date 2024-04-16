conv = ConvBp()
norms = [
            nothing,
            BatchNormBp(),
            GroupNormBp(; groups = 2),
            InstanceNormBp()
        ]
#dw, revs = pres = [true false]
for n in norms, r in [true false], p in [true false], d in [true false]
    c = cyanotype(conv; norm=n, preactivation=p, revnorm=r, depthwise=d)
    layers = flatten_layers(make(c, 3, 8 => 16))
    m = Chain(layers...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = DoubleConvBp(; conv1=ConvBp(), conv2=ConvBp(; norm=BatchNormBp()))
model = Chain(make(dc, 3, (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

dc = DoubleConvBp(; conv1=ConvBp(), conv2=nothing)
model = Chain(make(dc, 3, (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

model = Chain(make(NConvBp(; convolutions=(ConvBp(), ConvBp(), ConvBp(), ConvBp())), 3, 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(NConvBp(; convolutions=(ConvBp(), ConvBp(), ConvBp(), ConvBp())), 3, (4, 8, 16))...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(NConvBp(; convolutions=(ConvBp(), ConvBp(), ConvBp(), ConvBp())), 3, 4)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)

model = Chain(make(HybridAtrouConvBp(), 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

hac = HybridAtrouConvBp()
model = Chain(make(hac, 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = PointwiseConvBp()
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = PointwiseConvBp(norm=BatchNormBp())
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = PointwiseConvBp(activation=swish, init=Flux.glorot_normal)
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = ChannelExpansionConvBp(expansion=2)
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = ChannelExpansionConvBp(expansion=2, norm=BatchNormBp())
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = ChannelExpansionConvBp(expansion=1, norm=BatchNormBp())
@test make(bp, 4) == identity

bp = DepthwiseConvBp()
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = DepthwiseConvBp(norm=BatchNormBp())
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = DepthwiseConvBp(norm=BatchNormBp(), depthwise=false, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = MbConvBp(stride=2, ch_expansion=6, se_reduction=4)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (16, 16, 4, 16)

bp = MbConvBp(stride=1, ch_expansion=6, se_reduction=4, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)

se = SqueezeExcitationBp(; reduction = 4)
layers = Chain(flatten_layers(make(se, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

ca = ChannelAttentionBp(reduction=2)
layers = Chain(flatten_layers(make(ca, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

sa = SpatialAttentionBp()
layers = Chain(flatten_layers(make(sa, 3))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

cbam = CBAMBp(reduction=2)
layers = Chain(flatten_layers(make(cbam, 3, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

bp = FusedMbConvBp(stride=2, ch_expansion=6)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (16, 16, 4, 16)

bp = FusedMbConvBp(stride=1, ch_expansion=6, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)

bp = AxialDWConvBp(activation=gelu)
model = Chain(make(bp, 7, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 4)) == (32, 32, 16, 4)
