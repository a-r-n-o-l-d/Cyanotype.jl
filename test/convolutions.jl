conv = BpConv()
norms = [
            nothing,
            BpBatchNorm(),
            BpGroupNorm(; groups = 2),
            BpInstanceNorm()
        ]
#dw, revs = pres = [true false]
for n in norms, r in [true false], p in [true false], d in [true false]
    c = cyanotype(conv; normalization=n, preactivation=p, revnorm=r, depthwise=d)
    layers = flatten_layers(make(c, 3, 8 => 16))
    m = Chain(layers...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = BpDConv(; conv1=BpConv(), conv2=BpConv(; normalization=BpBatchNorm()))

model = Chain(make(dc, 3, (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

model = Chain(make(BpNConv(; convolution=BpConv(), nrepeat=3), 3, 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(BpHybridAtrouConv(), 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

hac = BpHybridAtrouConv()
model = Chain(make(hac, 3, 4 => 16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = BpPointwiseConv()
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = BpPointwiseConv(normalization=BpBatchNorm())
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = BpPointwiseConv(activation=swish, init=Flux.glorot_normal)
model = Chain(make(bp, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = BpChannelExpansionConv(expansion=2)
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = BpChannelExpansionConv(expansion=2, normalization=BpBatchNorm())
model = Chain(make(bp, 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 8, 16)
bp = BpChannelExpansionConv(expansion=1, normalization=BpBatchNorm())
@test make(bp, 4) == identity

bp = BpDepthwiseConv()
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = BpDepthwiseConv(normalization=BpBatchNorm())
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
bp = BpDepthwiseConv(normalization=BpBatchNorm(), depthwise=false, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 16) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

bp = BpMBConv(stride=2, ch_expansion=6, se_reduction=4)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (16, 16, 4, 16)

bp = BpMBConv(stride=1, ch_expansion=6, se_reduction=4, init=Flux.glorot_normal)
model = Chain(make(bp, 3, 4 => 4) |> flatten_layers)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 4, 16)

se = BpSqueezeExcitation(; reduction = 4)
layers = Chain(flatten_layers(make(se, 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)
