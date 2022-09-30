CyConv(; normalization = CyIdentityNorm()) |> println

CyConv(; normalization = CyBatchNorm()) |> println

methods(CyConv) |> println

fieldnames(CyConv) |> println

methods(CyBatchNorm) |> println

fieldnames(CyBatchNorm) |> println

CyConv().volumetric |> println

println(build(3, 16=>64, CyConv()))

c = CyConv(; normalization = CyBatchNorm())
println(build(3, 16=>64, c))

c = CyConv(; normalization = CyBatchNorm(), pre_activation = true)
println(Chain(build(3, 16=>64, c)...))

c = CyConv(; normalization = CyBatchNorm(), pre_activation = true, reverse_norm = true)
println(Chain(build(3, 16=>64, c)...))
