var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Cyanotype","category":"page"},{"location":"#Cyanotype","page":"Home","title":"Cyanotype","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Cyanotype.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Cyanotype]","category":"page"},{"location":"#Cyanotype.BatchNormBp","page":"Home","title":"Cyanotype.BatchNormBp","text":"Wraps a Flux.Batchnorm\n\nKeyword arguments:\n\nactivation: activation function (default relu)\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default true)\ntrack_stats: see track_stats (default true)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.ConvBp","page":"Home","title":"Cyanotype.ConvBp","text":"ConvBp(; kwargs)\n\nA cyanotype blueprint describing a convolutionnal module or layer depending om the value of normalization argument.\n\nKeyword arguments:\n\nvolumetric: indicates a building process for three-dimensionnal data (default false)\nnormalization:\nreverse_norm:\npre_activation:\nuse_bias:\ninit: see init (default glorot_uniform)\npad: see pad (default Flux.SamePad())\ndilation: see dilation (default 1)\ngroups: see groups (default 1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.DoubleConvBp","page":"Home","title":"Cyanotype.DoubleConvBp","text":"DoubleConvBp(; kwargs)\n\nDescribes a convolutionnal module formed by two successive convolutionnal modules.\n\nKeyword arguments:\n\nvolumetric: indicates a building process for three-dimensionnal data (default false)\nconvolution1 is not documented\nconvolution2 is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.GroupNormBp","page":"Home","title":"Cyanotype.GroupNormBp","text":"GroupNormBp(; kwargs...)\n\nDescribes a building process for a Groupnorm layer. make(channels, bp::CyGroupNorm)\n\nKeyword arguments:\n\nactivation: activation function (default relu)\ngroups: the number of groups passed to GroupNorm constructor\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default true)\ntrack_stats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.HybridAtrouConvBp","page":"Home","title":"Cyanotype.HybridAtrouConvBp","text":"aka Hybrid Dilated Convolution paper example example\n\nKeyword arguments:\n\ndilation_rates is not documented\nconvolution is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.InstanceNormBp","page":"Home","title":"Cyanotype.InstanceNormBp","text":"InstanceNormBp(; kwargs...)\n\nDescribes a building process for a InstanceNorm layer. make(channels, bp::CyInstanceNorm)\n\nKeyword arguments:\n\nactivation: activation function (default relu)\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default false)\ntrack_stats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.KwargsMapping","page":"Home","title":"Cyanotype.KwargsMapping","text":"KwargsMapping(; flux_function = :notflux, field_names = (), flux_kwargs = (),\n                field_types = (), def_values = ())\n\nDefine a mapping of keyword arguments mapping to interface a blueprint with a Flux function or constructor.\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.NConvBp","page":"Home","title":"Cyanotype.NConvBp","text":"Template describing a module with N NConvBp repeated.\n\nKeyword arguments:\n\nconvolution is not documented\nnrepeat is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.NoNormBp","page":"Home","title":"Cyanotype.NoNormBp","text":"NoNormBp()\n\nKeyword arguments:\n\nactivation: activation function (default Flux.relu)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.cyanotype","page":"Home","title":"Cyanotype.cyanotype","text":"cyanotype(bp::AbstractBlueprint; kwargs...)\n\nCreates a new blueprint from bp with the modifications defined by kwargs. This method is automatically generated by the @cyanotype macro during the process of defining a blueprint.\n\n\n\n\n\n","category":"function"},{"location":"#Cyanotype.make","page":"Home","title":"Cyanotype.make","text":"\n\n\n\n","category":"function"},{"location":"#Cyanotype.spread-Tuple{Any}","page":"Home","title":"Cyanotype.spread","text":"spread(bp; kwargs...)\n\nReturn a new cyanotype blueprint with kwargs spreaded over all fields of bp. This function allows to modify all nested cyanotypes at once.\n\n# Nice printing of large objects\nusing GarishPrint\n\n# Create a blueprint for a Hybrid A-trou Convolution module\nhac = CyHybridAtrouConv();\n# By default activation functions are relu\npprint(hac)\n\n# Create a blueprint for a double convolution module with the second convolution as a usual\n# convolutionnal layer\nconv2 = CyDoubleConv(; convolution1 = hac, convolution2 = CyConv());\n# By default activation functions are relu\npprint(conv2)\n\n# Now let change all activation functions from relu to leakyrelu\nspread(conv2; activation = leakyrelu) |> pprint\n\n\n\n\n\n","category":"method"},{"location":"#Cyanotype.@cyanotype-Tuple{Any, Any}","page":"Home","title":"Cyanotype.@cyanotype","text":"@cyanotype(doc, expr)\n@cyanotype(kmap, doc, expr)\n\nDefines a blueprint DataType with documentation doc and a struct declaration defined in expr. If the blueprint directly refers to a Flux function or constructor, kmap is the name of keyword arguments mapping. If there is some fields documentation in expr, it is automatically appended to doc.\n\nAutomatically generated functions:\n\nFooBluePrint(; kwargs...): keyword argument constructor for FooBluePrint\nmapping(::FooBluePrint): return, if defined, the mapping kmap\ncyanotype(bp::FooBluePrint)\nkwargs(bp::FooBluePrint): return, if defined, a Dict with the keyword arguments for\n\nthe Flux function or constructor, it can be used as follow: flux_function(arg1, arg2; kwargs(bp)...)\n\nExample:\n\nusing Cyanotype: @cyanotype\n\n@cyanotype (\n\"\"\"\nA FooBluePrint as example.\n\"\"\"\n) (\nstruct FooBluePrint{A<:Function}\n    \"\"\"`activation`: activation function\"\"\"\n    activation::A = relu\nend\n)\n\nFor the keyword arguments mapping usage, see KwargsMapping documentation.\n\n\n\n\n\n","category":"macro"}]
}
