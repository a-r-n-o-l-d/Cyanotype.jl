abstract type AbstractCyConv <: AbstractCyanotype end

const CyPad = Union{SamePad,Int}

register_mapping!(:convmap=>KwargsMapping(;
    flux_function  = :Conv,
    field_names    = (:init,               :pad,      :dilation, :groups),
    flux_kwargs    = (:init,               :pad,      :dilation, :groups),
    field_types    = (:I,                  :P,        :Int,      :Int),
    def_values     = (Flux.glorot_uniform, SamePad(), 1,         1)
))

@cyanotype convmap """

""" struct CyConv{N<:AbstractCyNorm,A<:Function,I<:Function,P<:CyPad} <: AbstractCyConv
    @activation(relu)
    @volumetric
    """
    `normalization`:
    """
    normalization::N = CyIdentityNorm()
    """
    `reverse_norm`:
    """
    reverse_norm::Bool = false
    """
    `pre_activation`:
    """
    pre_activation::Bool = false
    """
    `use_bias`:
    """
    use_bias::Bool = normalization isa CyIdentityNorm
end

function build(ksize, channels, cy::CyConv)
    k = cy.volumetric ? (ksize, ksize, ksize) : (ksize, ksize)
    #kwargs = curate(cy)
    layers = []
    # A regular convolutionnal layer
    if cy.normalization isa CyIdentityNorm
        #layers = [Conv(k, channels, cy.activation; kwargs...)]
        push!(layers, Conv(k, channels, cy.activation; kwargs(cy)...))
    # Add a normalization layer
    else
        in_chs, out_chs = channels
        # Normalization first
        if cy.reverse_norm
            # Activation before convolution ?
            if cy.pre_activation
                act_n = cy.activation
                act_c = identity
            else
                act_n = identity
                act_c = cy.activation
            end
            norm = cyanotype(cy.normalization; activation = act_n)
            conv = Conv(k, out_chs=>out_chs, act_c; bias = cy.use_bias, kwargs(cy)...)
            #layers = [build(in_chs, norm), conv]
            push!(layers, build(in_chs, norm), conv)
        # Convolution first
        else
            # Activation before convolution ?
            if cy.pre_activation
                # start by applying the activation function
                preact = cy.activation
                act_n = identity
            else
                # just for convenience, will be removed by flatten_layers
                preact = identity
                act_n = cy.activation
            end
            norm = cyanotype(cy.normalization; activation = act_n)
            conv = Conv(k, in_chs=>out_chs; bias = cy.use_bias, kwargs(cy)...)
            #layers = [preact, conv, build(out_chs, norm)]
            push!(layers, preact, conv, build(out_chs, norm))
        end
    end #|> flatten_layers
    # flatten and remove useless identity
    flatten_layers(layers)
end
