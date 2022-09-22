using Cyanotype
#using Flux
using Test
using Aqua
using JET

#Aqua.test_all(Cyanotype)


@testset verbose = true "Cyanotype.jl" begin
    @testset verbose = true "Code quality" begin
        @testset verbose = true "Aqua" begin
            #Aqua.test_all(Cyanotype) => ambiguities from
            Aqua.test_ambiguities(Cyanotype)
            Aqua.test_unbound_args(Cyanotype)
            Aqua.test_undefined_exports(Cyanotype)
            Aqua.test_project_extras(Cyanotype)
            Aqua.test_stale_deps(Cyanotype)
            Aqua.test_deps_compat(Cyanotype)
            Aqua.test_project_toml_formatting(Cyanotype)
        end
        @testset verbose = true "JET" begin
            jet_report = JET.report_package(Cyanotype)
            #@test string(jet_report) == "No errors detected\n"
        end
    end

    @testset "Config" begin
        include("config.jl")
    end
end
