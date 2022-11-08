se = SqueezeExciteBp(; reduction = 4)
layers = Chain(flatten_layers(make(se; channels = 16))...)
@test Flux.outputsize(layers, (32, 32, 16, 16)) == (32, 32, 16, 16)
