using JLD2, LinearAlgebra, TOML
length(ARGS)==3 || error("Usage: compare_gpu_results.jl main.jld2 candidate.jld2 report.toml")
a=JLD2.load(ARGS[1]); b=JLD2.load(ARGS[2])
@assert isempty(a["errors"]) a["errors"]
@assert isempty(b["errors"]) b["errors"]
@assert keys(a["results"]) == keys(b["results"])
exact=String[]; differences=Dict{String,Any}()
for key in sort!(collect(keys(a["results"])))
    x=a["results"][key]; y=b["results"][key]
    if isequal(x,y)
        push!(exact,key)
    elseif x isa AbstractArray && y isa AbstractArray
        @assert size(x)==size(y)
        differences[key]=Dict("relative_error"=>norm(y-x)/max(norm(x),eps()),
            "max_absolute_error"=>maximum(abs.(y-x)))
    else
        differences[key]=Dict("main"=>string(x),"candidate"=>string(y))
    end
end
report=Dict("exact_count"=>length(exact),"total_count"=>length(a["results"]),
    "exact"=>exact,"differences"=>differences,
    "main_metadata"=>Dict(string(k)=>string(v) for (k,v) in pairs(a["metadata"])),
    "candidate_metadata"=>Dict(string(k)=>string(v) for (k,v) in pairs(b["metadata"])))
open(ARGS[3],"w") do io
    TOML.print(io,report;sorted=true)
end
println("Exact GPU outputs: ",length(exact),"/",length(a["results"]))
for key in sort!(collect(keys(differences)))
    println("DIFFERENCE ",key," ",differences[key])
end
isempty(differences) || error("GPU outputs differ; review report before integration.")
