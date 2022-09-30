abstract type AbstractCyConv <: AbstractCyano end

register_mapping!(:convmap=>KwargsMapping(;
flux_function  = :Conv,
field_names    = (:init,          :pad,                :dilation, :groups),
flux_names     = (:init,          :pad,                :dilation, :groups),
field_types    = (:Function,      :CyPad, :Int,      :Int),
field_defaults = (Flux.glorot_uniform, 0,                  1,        1),
additional_doc =
"""pouet pouet"""
))

@cyano convmap struct CyConv{N<:AbstractCyanoNorm} <: AbstractCyConv
    @activation(relu)
    @volumetric
    """
    `normalization`:
    """
    normalization::N = CyanoIdentityNorm()
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
    use_bias::Bool = normalization isa CyanoIdentityNorm
end

#=
reverse_norm => mets Batchnorm avant Conv
pre_activation => mets relu avant Conv

Si reverse_norm
    Si pre_activation
        BatchNorm(in_chs, relu), Conv(k, out_chs=>out_chs, identity, nobias)
    Sinon
        BatchNorm(in_chs, identity), Conv(k, out_chs=>out_chs, relu, nobias)
Sinon
    Si pre_activation
        relu, Conv(k, in_chs=>out_chs, identity, nobias), BatchNorm(out_chs, identity)
    Sinon
        Conv(k, in_chs=>out_chs, identity, nobias), BatchNorm(out_chs, relu)
=#
function build(ksize, channels, cy::CyConv)
    k = cy.volumetric ? (ksize, ksize, ksize) : (ksize, ksize)
    kwargs = curate(cy)
    if cy.normalization isa CyanoIdentityNorm
        layers = Conv(k, channels, cy.activation; kwargs...)
    else
        in_chs, out_chs = channels
        if cy.reverse_norm
            if cy.pre_activation
                act_n = cy.activation
                act_c = identity
            else
                act_n = identity
                act_c = cy.activation
            end
            norm = new_cyanotype(cy.normalization; activation = act_n)
            conv = Conv(k, out_chs=>out_chs, act_c; bias = cy.use_bias, kwargs...)
            layers = [build(in_chs, norm), conv]
        else
            if cy.pre_activation
                preact = cy.activation
                act_n = identity
            else
                preact = identity
                act_n = cy.activation
            end
            norm = new_cyanotype(cy.normalization; activation = act_n)
            conv = Conv(k, in_chs=>out_chs; bias = cy.use_bias, kwargs...)
            layers = [preact, conv, build(out_chs, norm)]
        end
    end #|> flatten_layers
    flatten_layers(layers)
end
