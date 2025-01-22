norm = BatchNormBp()
baseconv = DoubleConvBp(conv1=ConvBp(), conv2=ConvBp())
ups = PixelShuffleUpsamplerBp()
dws = MaxDownsamplerBp()
dec = UDecoderBp(conv=baseconv, up=ups)
enc = UEncoderBp(conv=DoubleConvBp(conv1=ConvBp(stride=2), conv2=ConvBp()), down=nothing)
bdg = UBridgeBp(conv=baseconv, down=dws, up=ups)
top = PixelClassifierBp(nclasses=4)
unet = UNetBp(enc=enc, dec=dec, bdg=bdg, top=top, expn=2, bwidth=16)
m = make(unet)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)

p = ResCBAMBp(reduc=4)
unet = UNetBp(enc=enc, dec=dec, bdg=bdg, top=top, expn=2, bwidth=16, path=p)
m = make(unet, 3, 3)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)

p = ResCBAMBp(reduc=4)
enc = UEncoderBp(conv=DoubleConvBp(conv1=ConvBp(stride=1), conv2=ConvBp()), down=MeanMaxDownsamplerBp())
dec = UDecoderBp(conv=baseconv, up=ConvTransposeUpsamplerBp())
unet = UNetBp(enc=enc, dec=dec, bdg=bdg, top=top, expn=2, bwidth=16, path=p, stem=ConvBp())
m = make(unet, 3, 3)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)

kfunc(level) = if level == 1
    7
else
    3
end

unet = UNet2Bp(enc=enc, dec=dec, bdg=bdg, top=top, expn=2, ksize=kfunc, bwidth=16)
m = make(unet)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)

p = l -> ResCBAMBp(reduc=4)
top=PixelMapBp(nmaps=2)
unet = UNet2Bp(enc=enc, dec=dec, bdg=bdg, top=top, expn=2, ksize=kfunc, bwidth=16, path=p)
m = make(unet, 3, 3)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 2, 2)

m = make(EfficientUNetBp())
@test m(rand(Float32, (32, 32, 3, 2))) |> size == (32, 32, 32, 2)
