sp = MeanMaxDownsamplerBp()
m = Chain(make(sp, 4)...)
@test m(ones(Float32, 4, 4, 4, 2)) |> size == (2, 2, 4, 2)

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

sp = LinearUpsamplerBp(vol=true)
m = make(sp)
@test m(ones(4, 4, 4, 4, 2)) |> size == (8, 8, 8, 4, 2)

sp = ConvTransposeUpsamplerBp()
m = Chain(make(sp, 4))
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = PixelShuffleUpsamplerBp()
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 2)) == (8, 8, 4, 2)

sp = PixelShuffleUpsamplerBp(vol=true)
m = Chain(make(sp, 4)...)
@test Flux.outputsize(m, (4, 4, 4, 4, 2)) == (8, 8, 8, 4, 2)
