var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Cyanotype","category":"page"},{"location":"#Cyanotype","page":"Home","title":"Cyanotype","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Cyanotype.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Cyanotype]","category":"page"},{"location":"#Cyanotype.BatchNormBp","page":"Home","title":"Cyanotype.BatchNormBp","text":"Wraps a Flux.Batchnorm\n\nKeyword arguments:\n\nactivation: activation function (default identity)\ninitshift: see initβ (default zeros32)\ninitscale: see initγ (default ones32)\naffine: see affine (default true)\ntrackstats: see track_stats (default true)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CBAMBp","page":"Home","title":"Cyanotype.CBAMBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.ChannelAttentionBp","page":"Home","title":"Cyanotype.ChannelAttentionBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.ChannelExpansionConvBp","page":"Home","title":"Cyanotype.ChannelExpansionConvBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.ConvBp","page":"Home","title":"Cyanotype.ConvBp","text":"ConvBp(; kwargs...)\n\nA cyanotype blueprint describing a convolutionnal module or layer depending om the value of normalization argument.\n\nKeyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nactivation: activation function (default identity)\nnormalization:\ndepthwise:\nrevnorm:\npreactivation:\nbias:\nstride: see stride (default 1)\npad: see pad (default Flux.SamePad())\ndilation: see dilation (default 1)\ngroups: see groups (default 1)\ninit: see init (default glorot_uniform)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.ConvTransposeUpsamplerBp","page":"Home","title":"Cyanotype.ConvTransposeUpsamplerBp","text":"Keyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nscale is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.DepthwiseConvBp","page":"Home","title":"Cyanotype.DepthwiseConvBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.DoubleConvBp","page":"Home","title":"Cyanotype.DoubleConvBp","text":"DoubleConvBp(; kwargs)\n\nDescribes a convolutionnal module formed by two successive convolutionnal modules.\n\nKeyword arguments:\n\nconv1 is not documented\nconv2 is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.EfficientNetBp","page":"Home","title":"Cyanotype.EfficientNetBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.EfficientNetStageBp","page":"Home","title":"Cyanotype.EfficientNetStageBp","text":"Keyword arguments:\n\nksize is not documented\noutchannels is not documented\nnrepeat is not documented\nconvolution is not documented\nwidthscaling is not documented\ndepthscaling is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.FusedMbConvBp","page":"Home","title":"Cyanotype.FusedMbConvBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.GroupNormBp","page":"Home","title":"Cyanotype.GroupNormBp","text":"GroupNormBp(; kwargs...)\n\nDescribes a building process for a Groupnorm layer. make(channels, bp::CyGroupNorm)\n\nKeyword arguments:\n\nactivation: activation function (default identity)\ngroups: the number of groups passed to GroupNorm\n\nconstructor\n\ninitshift: see initβ (default zeros32)\ninitscale: see initγ (default ones32)\naffine: see affine (default true)\ntrackstats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.HybridAtrouConvBp","page":"Home","title":"Cyanotype.HybridAtrouConvBp","text":"aka Hybrid Dilated Convolution paper example example\n\nKeyword arguments:\n\ndilation_rates is not documented\nconv is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.InstanceNormBp","page":"Home","title":"Cyanotype.InstanceNormBp","text":"InstanceNormBp(; kwargs...)\n\nDescribes a building process for a InstanceNorm layer. make(channels, bp::CyInstanceNorm)\n\nKeyword arguments:\n\nactivation: activation function (default identity)\ninitshift: see initβ (default zeros32)\ninitscale: see initγ (default ones32)\naffine: see affine (default false)\ntrackstats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.KwargsMapping","page":"Home","title":"Cyanotype.KwargsMapping","text":"KwargsMapping(; flux_function = :notflux, field_names = (), flargs = (),\n                ftypes = (), defval = ())\n\nDefine a mapping of keyword arguments to interface a blueprint with a Flux function or constructor.\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.LabelClassifierBp","page":"Home","title":"Cyanotype.LabelClassifierBp","text":"Keyword arguments:\n\nnclasses is not documented\ndropout is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.LinearUpsamplerBp","page":"Home","title":"Cyanotype.LinearUpsamplerBp","text":"Keyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nscale is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.MaxDownsamplerBp","page":"Home","title":"Cyanotype.MaxDownsamplerBp","text":"Keyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nwsize is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.MbConvBp","page":"Home","title":"Cyanotype.MbConvBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.MeanDownsamplerBp","page":"Home","title":"Cyanotype.MeanDownsamplerBp","text":"Keyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nwsize is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.NConvBp","page":"Home","title":"Cyanotype.NConvBp","text":"Template describing a module with N NConvBp repeated.\n\nKeyword arguments:\n\nconvolution is not documented\nnrepeat is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.NearestUpsamplerBp","page":"Home","title":"Cyanotype.NearestUpsamplerBp","text":"Keyword arguments:\n\nvolume: indicates a building process for three-dimensionnal data (default false)\nscale is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.PixelClassifierBp","page":"Home","title":"Cyanotype.PixelClassifierBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.PixelShuffleUpsamplerBp","page":"Home","title":"Cyanotype.PixelShuffleUpsamplerBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.PointwiseConvBp","page":"Home","title":"Cyanotype.PointwiseConvBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.SpatialAttentionBp","page":"Home","title":"Cyanotype.SpatialAttentionBp","text":"\n\n\n\n","category":"type"},{"location":"#Cyanotype.SqueezeExcitationBp","page":"Home","title":"Cyanotype.SqueezeExcitationBp","text":"https://arxiv.org/pdf/1709.01507.pdf\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.UBridgeBp","page":"Home","title":"Cyanotype.UBridgeBp","text":"Keyword arguments:\n\nconvolution is not documented\ndownsampler is not documented\nupsampler is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.UDecoderBp","page":"Home","title":"Cyanotype.UDecoderBp","text":"Keyword arguments:\n\nconvolution is not documented\nupsampler is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.UEncoderBp","page":"Home","title":"Cyanotype.UEncoderBp","text":"Keyword arguments:\n\nconvolution is not documented\ndownsampler is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.UNetBp","page":"Home","title":"Cyanotype.UNetBp","text":"Keyword arguments:\n\ninchannels is not documented\nnlevels is not documented\nbasewidth is not documented\nexpansion is not documented\nksize is not documented\nencoder is not documented\ndecoder is not documented\nbridge is not documented\nstem is not documented\npath is not documented\nhead is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.chcat-Tuple","page":"Home","title":"Cyanotype.chcat","text":"chcat(x...)\n\nConcatenate the image data along the dimension corresponding to the channels. Image data should be stored in WHCN order (width, height, channels, batch) or WHDCN (width, height, depth, channels, batch) in 3D context. Channels are assumed to be the penultimate dimension.\n\nExample\n\njulia> x1 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels\n\njulia> x2 = rand(32, 32, 4, 6); # a batch of 6 images (32x32) with 4 channels\n\njulia> chcat(x1, x2) |> size\n(32, 32, 8, 6)\n\n\n\n\n\n","category":"method"},{"location":"#Cyanotype.cyanotype","page":"Home","title":"Cyanotype.cyanotype","text":"cyanotype(bp::AbstractBlueprint; kwargs...)\n\nCreate a new blueprint from bp with the modifications defined by kwargs. This method is automatically generated by the @cyanotype macro during the process of defining a blueprint.\n\n\n\n\n\n","category":"function"},{"location":"#Cyanotype.make","page":"Home","title":"Cyanotype.make","text":"\n\n\n\n","category":"function"},{"location":"#Cyanotype.spread-Tuple{Any}","page":"Home","title":"Cyanotype.spread","text":"spread(bp; kwargs...)\n\nReturn a new blueprint with kwargs spreaded over all fields of bp. This function allows to modify all nested cyanotypes at once.\n\n# Nice printing of large objects\nusing GarishPrint\n\n# Create a blueprint for a Hybrid A-trou Convolution module\nhac = CyHybridAtrouConv();\n# By default activation functions are relu\npprint(hac)\n\n# Create a blueprint for a double convolution module with the second convolution as a usual\n# convolutionnal layer\nconv2 = CyDoubleConv(; convolution1 = hac, convolution2 = CyConv());\n# By default activation functions are relu\npprint(conv2)\n\n# Now let change all activation functions from relu to leakyrelu\nspread(conv2; activation = leakyrelu) |> pprint\n\n\n\n\n\n","category":"method"},{"location":"#Cyanotype.uchain-Tuple{}","page":"Home","title":"Cyanotype.uchain","text":"uchain(; encoders, decoders, bridge, connection, [paths])\n\nBuild a Chain with U-Net like architecture. encoders and decoders are vectors of encoding/decoding blocks, from top to bottom (see diagram below). bridge is the bottom part of U-Net  architecture. Each level of the U-Net is connected through a channel concatenation (◊ symbol in the diagram below).\n\nNotes :\n\nusually encoder unit starts with a 'MaxPool' to downsample image by 2, except the first\n\nlevel encoder.\n\nusually decoder unit ends with a 'ConvTranspose' to upsample image by 2, except the first\n\nlevel decoder.\n\n┌─────────┐                                                          ┌─────────┐\n│Encoder 1│                                                          │Decoder 1│\n└────┬────┘                                                          └─────────┘\n     │                                                                    ▴\n     │                                                                    │\n     ├───────────────────────────────────────────────────────────────────▸◊\n     │    ┌─────────┐                                     ┌─────────┐     │\n     └───▸│Encoder 2│                                     │Decoder 2├─────┘\n          └────┬────┘                                     └─────────┘\n               │                                               ▴\n               │                                               │\n               ├──────────────────────────────────────────────▸◊\n               │     ┌─────────┐               ┌─────────┐     │\n               └────▸│Encoder 3│               │Decoder 3├─────┘\n                     └────┬────┘               └─────────┘\n                          │                         ▴\n                          │                         │\n                          ├────────────────────────▸◊\n                          │       ┌─────────┐       │\n                          └──────▸│ Bridge  ├───────┘\n                                  └─────────┘\n\nSee also chcat.\n\n\n\n\n\n","category":"method"},{"location":"#Cyanotype.@cyanotype-Tuple{Any}","page":"Home","title":"Cyanotype.@cyanotype","text":"@cyanotype(expr)\n@cyanotype begin\n    [kmap]\n    [doc]\n    expr\nend\n\nDefines a blueprint DataType with documentation doc and a struct declaration defined in expr. If the blueprint directly refers to a Flux function or constructor, kmap is the name of keyword arguments mapping. If there is some fields documentation in expr, it is automatically appended to doc.\n\nAutomatically generated functions:\n\nFooBluePrint(; kwargs...): keyword argument constructor for FooBluePrint\nmapping(::FooBluePrint): return, if defined, the mapping kmap\ncyanotype(bp::FooBluePrint; kwargs...)\nkwargs(bp::FooBluePrint): return, if defined, a Dict with the keyword arguments for\n\nthe Flux function or constructor, it can be used as follow: flux_function(arg1, arg2; kwargs(bp)...)\n\nExample:\n\nusing Cyanotype: @cyanotype\n\n@cyanotype begin\n    \"\"\"\n    A FooBlueprint as example.\n    \"\"\"\n    struct FooBlueprint{A<:Function}\n        \"\"\"`activation`: activation function\"\"\"\n        activation::A = relu\n    end\nend\n\nFor the keyword arguments mapping usage, see KwargsMapping documentation.\n\n\n\n\n\n","category":"macro"}]
}
