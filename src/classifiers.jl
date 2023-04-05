abstract type AbstractClassifier end

@cyanotype begin
    """
    """
    struct PixelClassifierBp <: AbstractClassifier
        @volume
        nclasses::Int
    end
end

function make(bp::PixelClassifierBp, channels)
    k = genk(1, bp.volume)
    if bp.nclasses > 2
        [Conv(k, channels=>bp.nclasses), x -> softmax(x; dims = length(k))]
    else #if bp.nclasses == 2
        [Conv(k, channels=>1, sigmoid)]
    end #else Conv(k, channels=>1, sigmoid)
end
