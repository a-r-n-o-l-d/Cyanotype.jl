conv = ConvBp()
norms = [
            nothing,
            BatchNormBp(),
            GroupNormBp(; groups = 2),
            InstanceNormBp()
        ]
#dw, revs = pres = [true false]
for n in norms, r in [true false], p in [true false], d in [true false]
    c = cyanotype(conv; normalization=n, preactivation=p, revnorm=r, depthwise=d)
    layers = flatten_layers(make(c, 3, 8 => 16))
    m = Chain(layers...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = DoubleConvBp(; conv1=ConvBp(), conv2=ConvBp(; normalization=BatchNormBp()))

model = Chain(make(dc, 3, (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

model = Chain(make(NConvBp(; convolution=ConvBp(), nrepeat=3), 3, 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(HybridAtrouConvBp(), 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

hac = HybridAtrouConvBp()
model = Chain(make(hac, 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = PointwiseConvBp()
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = PointwiseConvBp(normalization=BatchNormBp())
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = PointwiseConvBp(activation=swish, init=Flux.glorot_normal)
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = ChannelExpansionConvBp(expansion=2)
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = ChannelExpansionConvBp(expansion=2, normalization=BatchNormBp())
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = ChannelExpansionConvBp(expansion=1, normalization=BatchNormBp())
@test make(bp, 4) == identity

bp = DepthwiseConvBp()
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = DepthwiseConvBp(normalization=BatchNormBp())
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = DepthwiseConvBp(normalization=BatchNormBp(), depthwise=false, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = BpMbConv(stride=2, ch_expansion=6, se_reduction=4)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (16, 16, 4, 16)

bp = BpMbConv(stride=1, ch_expansion=6, se_reduction=4, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)

se = BpSqueezeExcitation(; reduction = 4)
layers = Chain(flatten_layers(make(se, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

ca = BpChannelAttention(reduction=2)
layers = Chain(flatten_layers(make(ca, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

sa = BpSpatialAttention()
layers = Chain(flatten_layers(make(sa, 3))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

cbam = BpCBAM(reduction=2)
layers = Chain(flatten_layers(make(cbam, 3, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)

bp = BpFusedMbConv(stride=2, ch_expansion=6)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (16, 16, 4, 16)

bp = BpFusedMbConv(stride=1, ch_expansion=6, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)
