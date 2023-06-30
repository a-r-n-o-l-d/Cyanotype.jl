#ResidualBp
@cyanotype begin
    struct ResidualConvBp{C<:AbstractConvBp,F<:Function}
        convolution::C
        connector::F = +
    end
end

make(bp::ResidualConvBp, ksize, channels) = SkipConnection(make(bp.convolution, ksize, channels), bp.connector)
