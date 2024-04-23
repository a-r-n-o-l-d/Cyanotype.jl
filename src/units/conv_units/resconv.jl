#ResidualBp
@cyanotype begin
    struct ResidualConvBp <: AbstractConvBp
        conv
        connector = +
    end
end

make(bp::ResidualConvBp, ksize, channels) = SkipConnection(
    Chain(flatten_layers(make(bp.conv, ksize, channels))...),
    bp.connector
)
