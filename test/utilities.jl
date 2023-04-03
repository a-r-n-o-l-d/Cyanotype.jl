hac = BpHybridAtrouConv()
hac2 = spread(hac; activation = leakyrelu)
@test hac2.conv.normalization.activation == leakyrelu
