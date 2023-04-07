sp = BpMaxDown()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (2, 2, 4, 2)

sp = BpMeanDown()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (2, 2, 4, 2)

sp = BpNearestUp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (8, 8, 4, 2)

sp = BpLinearUp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (8, 8, 4, 2)

sp = BpLinearUp(volume=true)
m = make(sp)
@test m(ones(4, 4, 4, 4, 2)) |> size == (8, 8, 8, 4, 2)

sp = BpConvTransposeUp()
m = Chain(make(sp, 4))
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = BpPixelShuffleUp()
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = BpPixelShuffleUp(volume=true)
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 4, 2)) == (8, 8, 8, 4, 2)
