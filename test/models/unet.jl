norm = BatchNormBp()
baseconv = DoubleConvBp(conv1=ConvBp(), conv2=ConvBp())
ups = PixelShuffleUpsamplerBp()
dws = MaxDownsamplerBp()
dec = UDecoderBp(conv=baseconv, upsampler=ups)
enc = UEncoderBp(conv=DoubleConvBp(conv1=ConvBp(stride=2), conv2=ConvBp()), downsampler=nothing)
bdg = UBridgeBp(conv=baseconv, downsampler=dws, upsampler=ups)
top = PixelClassifierBp(nclasses=4)
unet = UNetBp(encoder=enc, decoder=dec, bridge=bdg, top=top, expansion=2, basewidth=16)
m = make(unet)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)

kfunc(level) = if level == 1
    7
else
    3
end

unet = UNet2Bp(encoder=enc, decoder=dec, bridge=bdg, top=top, expansion=2, ksize=kfunc, basewidth=16)
m = make(unet)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)
