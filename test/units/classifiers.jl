pc = BpPixelClassifier(; nclasses=1)
m = Chain(make(pc, 16)...)
@test Flux.outputsize(m, (4, 4, 16, 16)) == (4, 4, 1, 16)

pc = BpPixelClassifier(; nclasses=4)
m = Chain(make(pc, 16)...)
@test Flux.outputsize(m, (4, 4, 16, 16)) == (4, 4, 4, 16)
