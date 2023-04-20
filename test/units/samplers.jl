sp = MaxDownsamplerBp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (2, 2, 4, 2)

sp = MeanDownsamplerBp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (2, 2, 4, 2)

sp = NearestUpsamplerBp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (8, 8, 4, 2)

sp = LinearUpsamplerBp()
m = make(sp)
@test m(ones(4, 4, 4, 2)) |> size == (8, 8, 4, 2)

sp = LinearUpsamplerBp(volume=true)
m = make(sp)
@test m(ones(4, 4, 4, 4, 2)) |> size == (8, 8, 8, 4, 2)

sp = BpConvTransposeUpsampler()
m = Chain(make(sp, 4))
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = BpPixelShuffleUpsampler()
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = BpPixelShuffleUpsampler(volume=true)
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 4, 2)) == (8, 8, 8, 4, 2)
