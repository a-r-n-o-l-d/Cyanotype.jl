conv = ConvBp()
norms = [NoNormBp(), BatchNormBp(), GroupNormBp(; groups = 2), InstanceNormBp()]
revs = pres = [true, false]
for n in norms, r in revs, p in pres
    c = cyanotype(conv; normalization = n, pre_activation = p, reverse_norm = r)
    m = Chain(make(c; ksize = 3, channels = 8=>16)...)
    @test Flux.outputsize(m, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = DoubleConvBp(; convolution1 = ConvBp(), convolution2 = ConvBp(; normalization = BatchNormBp()))

model = Chain(make(dc; ksize = 3, channels = (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)

model = Chain(make(NConvBp(; convolution = ConvBp(), nrepeat = 3), 3, 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)

model = Chain(make(HybridAtrouConvBp(); ksize = 3, channels = 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)


hac = HybridAtrouConvBp()
model = Chain(make(hac; ksize = 3, channels = 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
