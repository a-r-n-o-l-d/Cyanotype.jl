using Cyanotype
using Flux
using Test
using Aqua

@testset verbose = true "Cyanotype.jl" begin

    @testset verbose = true "Code quality" begin
        @testset verbose = true "Aqua" begin
            #Aqua.test_all(Cyanotype) #  => ambiguities from Flux, Zygote, StatsBase
            Aqua.test_ambiguities(Cyanotype)
            Aqua.test_unbound_args(Cyanotype)
            Aqua.test_undefined_exports(Cyanotype)
            Aqua.test_project_extras(Cyanotype)
            Aqua.test_stale_deps(Cyanotype)
            Aqua.test_deps_compat(Cyanotype)
            Aqua.test_project_toml_formatting(Cyanotype)
        end
    end

    @testset verbose = true "utilities" begin
        include("utilities.jl")
    end

    @testset verbose = true "Cyanotype" begin
        include("cyanotype.jl")
    end

    @testset verbose = true "Normalizations" begin
        include("normalizations.jl")
    end

    @testset verbose = true "Convolutions" begin
        include("convolutions.jl")
    end

    @testset verbose = true "Samplers" begin
        include("samplers.jl")
    end

    @testset verbose = true "Classifiers" begin
        include("classifiers.jl")
    end

    @testset verbose = true "UNet" begin
        include("uchain.jl")
        include("unet.jl")
    end
end
