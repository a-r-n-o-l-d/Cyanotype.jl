var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = Cyanotype","category":"page"},{"location":"#Cyanotype","page":"Home","title":"Cyanotype","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Cyanotype.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Cyanotype]","category":"page"},{"location":"#Cyanotype.CyBatchNorm","page":"Home","title":"Cyanotype.CyBatchNorm","text":"Wraps a Flux.Batchnorm\n\nKeyword arguments:\n\nactivation: activation function (default Flux.relu)\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default true)\ntrack_stats: see track_stats (default true)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyConv","page":"Home","title":"Cyanotype.CyConv","text":"CyConv(; kwargs)\n\nDescribes a convolutionnal module or layer depending om the value of normalization argument.\n\nKeyword arguments:\n\nactivation: activation function (default relu)\nvolumetric: indicates a building process for three-dimensionnal data (default false)\nnormalization:\nreverse_norm:\npre_activation:\nuse_bias:\ninit: see init (default glorot_uniform)\npad: see pad (default Flux.SamePad())\ndilation: see dilation (default 1)\ngroups: see groups (default 1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyDoubleConv","page":"Home","title":"Cyanotype.CyDoubleConv","text":"CyDoubleConv(; kwargs)\n\nDescribes a convolutionnal module formed by two successive convolutionnal modules.\n\nKeyword arguments:\n\nvolumetric: indicates a building process for three-dimensionnal data (default false)\nconvolution1 is not documented\nconvolution2 is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyGroupNorm","page":"Home","title":"Cyanotype.CyGroupNorm","text":"CyGroupNorm(; kwargs...)\n\nDescribes a building process for a Groupnorm layer. build(channels, cy::CyGroupNorm)\n\nKeyword arguments:\n\nactivation: activation function (default relu)\ngroups: the number of groups passed to GroupNorm constructor\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default true)\ntrack_stats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyHybridAtrouConv","page":"Home","title":"Cyanotype.CyHybridAtrouConv","text":"aka Hybrid Dilated Convolution paper example example\n\nKeyword arguments:\n\ndilation_rates is not documented\nconvolution is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyInstanceNorm","page":"Home","title":"Cyanotype.CyInstanceNorm","text":"CyInstanceNorm(; kwargs...)\n\nDescribes a building process for a InstanceNorm layer. build(channels, cy::CyInstanceNorm)\n\nKeyword arguments:\n\nactivation: activation function (default relu)\ninit_shift: see initβ (default zeros32)\ninit_scale: see initγ (default ones32)\naffine: see affine (default true)\ntrack_stats: see track_stats (default false)\nepsilon: see ϵ (default 1.0e-5)\nmomentum: see momentum (default 0.1)\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyNConv","page":"Home","title":"Cyanotype.CyNConv","text":"Template describing a module with N CyConv repeated.\n\nKeyword arguments:\n\nconvolution is not documented\nnrepeat is not documented\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.CyNoNorm","page":"Home","title":"Cyanotype.CyNoNorm","text":"CyNoNorm()\n\nTagging struct indicating that no normalisation layer should be used in a building process.\n\nKeyword arguments:\n\n\n\n\n\n","category":"type"},{"location":"#Cyanotype.spread-Tuple{Any}","page":"Home","title":"Cyanotype.spread","text":"spread(cy; kwargs...)\n\nReturn a new cyanotype blueprint with kwargs spreaded over all fields of cy. This function allows to modify all nested cyanotypes at once.\n\n# Nice printing of large objects\nusing GarishPrint\n\n# Create a blueprint for a Hybrid A-trou Convolution module\nhac = CyHybridAtrouConv();\n# By default activation functions are relu\npprint(hac)\n\n# Create a blueprint for a double convolution module with the second convolution as a usual\n# convolutionnal layer\nconv2 = CyDoubleConv(; convolution1 = hac, convolution2 = CyConv());\n# By default activation functions are relu\npprint(conv2)\n\n# Now let change all activation functions from relu to leakyrelu\nspread(conv2; activation = leakyrelu) |> pprint\n\n\n\n\n\n","category":"method"}]
}
