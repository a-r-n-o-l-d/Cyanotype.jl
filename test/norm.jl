bn = BatchNormBp()
model = Chain(make(bn; channels = 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)

gn = GroupNormBp(; groups = 4)
model = Chain(make(gn; channels = 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)

in = InstanceNormBp()
model = Chain(make(in; channels = 16))
@test Flux.outputsize(model, (32, 32, 16, 16)) == (32, 32, 16, 16)
