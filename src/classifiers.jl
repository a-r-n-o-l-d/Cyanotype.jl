abstract type AbstractClassifier end

@cyanotype begin
    """
    """
    struct PixelClassifierBp <: AbstractClassifier
        @volumetric
        nclasses::Int
    end
end

function make(bp::PixelClassifierBp; channels)
    k = genk(1, bp.vol)
    if bp.nclasses > 2
        [Conv(k, channels=>bp.nclasses), x -> softmax(x; dims = length(k))]
    else #if bp.nclasses == 2
        [Conv(k, channels=>1, sigmoid)]
    end #else Conv(k, channels=>1, sigmoid)
end
