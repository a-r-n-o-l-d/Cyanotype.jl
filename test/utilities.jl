hac = HybridAtrouConvBp()
hac2 = spread(hac; activation=leakyrelu)
@test hac2.conv.norm.activation == leakyrelu
