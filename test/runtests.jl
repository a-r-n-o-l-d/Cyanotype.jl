using Cyanotype
using Cyanotype: CyFloat
#using Flux
using Test
#using Aqua

@testset verbose = true "Cyanotype.jl" begin

    #=@testset verbose = true "Code quality" begin
        @testset verbose = true "Aqua" begin
            #Aqua.test_all(Cyanotype) #  => ambiguities from Flux, Zygote, StatsBase
            #Aqua.test_ambiguities(Cyanotype)
            Aqua.test_unbound_args(Cyanotype)
            Aqua.test_undefined_exports(Cyanotype)
            Aqua.test_project_extras(Cyanotype)
            Aqua.test_stale_deps(Cyanotype)
            Aqua.test_deps_compat(Cyanotype)
            Aqua.test_project_toml_formatting(Cyanotype)
        end
    end=#

    @testset verbose = true "utilities" begin
        include("utilities.jl")
    end

    @testset verbose = true "Cyanotype" begin
        include("cyanotype.jl")
    end

    @testset verbose = true "Normalizations" begin
        include("units/normalizations.jl")
    end

    @testset verbose = true "Convolutions" begin
        include("units/convolutions.jl")
    end

    @testset verbose = true "Samplers" begin
        include("units/samplers.jl")
    end

    @testset verbose = true "Classifiers" begin
        include("units/classifiers.jl")
    end

    @testset verbose = true "UNet" begin
        include("models/uchain.jl")
        include("models/unet.jl")
    end

    @testset verbose = true "EfficientNet" begin
        include("models/efficientnet.jl")
    end

end
