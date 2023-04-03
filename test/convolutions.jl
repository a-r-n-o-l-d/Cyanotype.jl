conv = BpConv()
norms = [
            nothing, #BpNoNorm()
            BpBatchNorm(),
            BpGroupNorm(; groups = 2),
            BpInstanceNorm()
        ]
dw, revs = pres = [true, false]
for n in norms, r in revs, p in pres, d in dw
    c = cyanotype(conv; norm=n, preact=p, revnorm=r, depthwise=d)
    m = Chain(make(c; ksize=3, channels=8 => 16)...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = BpDConv(; conv1 = BpConv(),
                         conv2 = BpConv(; norm = BpBatchNorm()))

model = Chain(make(dc; ksize = 3, channels = (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

model = Chain(make(BpNConv(; convolution = BpConv(), nrepeat = 3); ksize = 3, channels = 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(BpHAConv(); ksize = 3, channels = 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)


hac = BpHAConv()
model = Chain(make(hac; ksize = 3, channels = 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
