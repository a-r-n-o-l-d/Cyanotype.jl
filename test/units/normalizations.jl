bn = BatchNormBp()
model = Chain(make(bn, 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)

gn = GroupNormBp(; groups = 4)
model = Chain(make(gn, 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)

in = InstanceNormBp()
model = Chain(make(in, 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)
