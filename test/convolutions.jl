conv = BpConv()
norms = [
            nothing, #BpNoNorm()
            BpBatchNorm(),
            BpGroupNorm(; groups = 2),
            BpInstanceNorm()
        ]
dw, revs = pres = [true, false]
for n in norms, r in revs, p in pres, d in dw
    c = cyanotype(conv; normalization=n, preactivation=p, revnorm=r, depthwise=d)
    layers = flatten_layers(make(c, 3, 8 => 16))
    m = Chain(layers...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = BpDConv(; conv1 = BpConv(),
                         conv2 = BpConv(; normalization=BpBatchNorm()))

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
