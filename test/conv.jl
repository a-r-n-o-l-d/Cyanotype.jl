CyConv(; normalization = CyanoIdentityNorm()) |> println

CyConv(; normalization = CyanoBatchNorm()) |> println

methods(CyConv) |> println

fieldnames(CyConv) |> println

methods(CyanoBatchNorm) |> println

fieldnames(CyanoBatchNorm) |> println

CyConv().volumetric |> println

println(build(3, 16=>64, CyConv()))

c = CyConv(; normalization = CyanoBatchNorm())
println(build(3, 16=>64, c))

c = CyConv(; normalization = CyanoBatchNorm(), pre_activation = true)
println(Chain(build(3, 16=>64, c)...))

c = CyConv(; normalization = CyanoBatchNorm(), pre_activation = true, reverse_norm = true)
println(Chain(build(3, 16=>64, c)...))
