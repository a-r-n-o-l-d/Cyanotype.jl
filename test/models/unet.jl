norm = BatchNormBp()
baseconv = DoubleConvBp(conv1=ConvBp(), conv2=ConvBp())
ups = BpPixelShuffleUpsampler()
dws = MaxDownsamplerBp()
dec = BpUDecoder(convolution=baseconv, upsampler=ups)
enc = BpUEncoder(convolution=DoubleConvBp(conv1=ConvBp(stride=2), conv2=ConvBp()), downsampler=nothing)
bdg = BpUBridge(convolution=baseconv, downsampler=dws, upsampler=ups)
h = PixelClassifierBp(nclasses=4)
unet = BpUNet(encoder=enc, decoder=dec, bridge=bdg, head=h, expansion=2, basewidth=16)
m = make(unet)
@test Flux.outputsize(m, (32, 32, 3, 2)) == (32, 32, 4, 2)
