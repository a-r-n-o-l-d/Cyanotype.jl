#=CyConv() |> println

CyConv(; normalization = CyBatchNorm()) |> println

methods(CyConv) |> println

fieldnames(CyConv) |> println

methods(CyBatchNorm) |> println

fieldnames(CyBatchNorm) |> println

CyConv().volumetric |> println

println(build(3, 16=>64, CyConv()))

c = CyConv(; normalization = CyBatchNorm())
println(build(3, 16=>64, c))

c = CyConv(; normalization = CyBatchNorm(), pre_activation = true)
println(Chain(build(3, 16=>64, c)...))

c = CyConv(; normalization = CyBatchNorm(), pre_activation = true, reverse_norm = true)
println(Chain(build(3, 16=>64, c)...))


conv = CyConv()

batch = rand(Float32, 32, 32, 3, 16)

model = Chain(build(3, 3=>4, conv)...)

@test Flux.outputsize(model, (32, 32, 3, 16)) == (32, 32, 4, 16)=#

conv = ConvBp()
norms = [NoNormBp(), BatchNormBp(), GroupNormBp(; groups = 2), InstanceNormBp()]
revs = pres = [true, false]
for n in norms, r in revs, p in pres
    c = cyanotype(conv; normalization = n, pre_activation = p, reverse_norm = r)
    model = Chain(make(c; ksize = 3, channels = 8=>16)...)
    #println(model)
    @test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 16, 16)
end

dc = DoubleConvBp(; convolution1 = ConvBp(), convolution2 = ConvBp(; normalization = BatchNormBp()))

model = Chain(make(dc; ksize = 3, channels = (8, 16, 32))...)
@test Flux.outputsize(model, (32, 32, 8, 16)) == (32, 32, 32, 16)


model = Chain(make(NConvBp(; convolution = ConvBp(), nrepeat = 3), 3, 4=>16)...)

@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)



model = Chain(make(HybridAtrouConvBp(), 3, 4=>16)...)

@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)


hac = HybridAtrouConvBp()

model = Chain(make(hac, 3, 4=>16)...)
@test Flux.outputsize(model, (32, 32, 4, 16)) == (32, 32, 16, 16)
