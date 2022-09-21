using Cyanotype
using Flux
using Test

@testset verbose = true "Cyanotype.jl" begin
    
    @testset "Config" begin
        include("config.jl")
    end
end
