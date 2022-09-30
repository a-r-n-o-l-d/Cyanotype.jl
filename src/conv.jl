register_mapping!(:convmap=>KwargsMapping(;
flux_function  = :Conv,
field_names    = (:init,          :pad,                :dilation, :groups),
flux_names     = (:init,          :pad,                :dilation, :groups),
field_types    = (:Function,      :CyPad, :Int,      :Int),
field_defaults = (Flux.glorot_uniform, 0,                  1,        1),
additional_doc =
"pouet pouet"
))

@cyano convmap struct CyConv{N<:AbstractCyanoNorm}
    @activation(relu)

    @volumetric

    """
    `normalization`
    """
    normalization::N = CyanoIdentityNorm()

    """
    `reverse_norm`
    """
    reverse_norm::Bool = false

    """
    `pre_activation`
    """
    pre_activation::Bool = false

    """
    `bias`
    """
    bias::Bool = normalization isa CyanoIdentityNorm
end
