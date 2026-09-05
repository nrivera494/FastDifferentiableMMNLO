using PulsePropagation, JLD2, LinearAlgebra, TOML
const PP = PulsePropagation
const root = @__DIR__
a = JLD2.load(joinpath(root,"main_results.jld2"),"results")
b = JLD2.load(joinpath(root,"integrated_results.jld2"),"results")
summary = Dict{String,Any}()
exact = String[]
changed = Dict{String,Any}()
errors = Dict{String,Any}()
for key in sort!(collect(keys(a)))
    if endswith(key,"/ERROR")
        errors[key] = Dict("main"=>a[key],"checkpointed"=>get(b,key,"missing"))
    elseif !haskey(b,key)
        changed[key] = Dict("status"=>"missing on checkpointed branch")
    elseif isequal(a[key],b[key])
        push!(exact,key)
    elseif a[key] isa AbstractArray && b[key] isa AbstractArray
        diff = norm(b[key] .- a[key])
        changed[key] = Dict("relative_error"=>diff/max(norm(a[key]),eps()),
            "max_absolute_error"=>maximum(abs.(b[key].-a[key])))
    else
        changed[key] = Dict("main"=>string(a[key]),"checkpointed"=>string(b[key]))
    end
end
summary["exact_count"] = length(exact)
summary["exact"] = exact
summary["changed"] = changed
summary["errors"] = errors
println("EXACT ", length(exact), " / ",length(a), "; ERRORS ",length(errors))
for key in sort!(collect(keys(changed)))
    println("CHANGED ",key," ", changed[key])
end
println("Notebook-facing numerical outputs exact: ",
    count(k -> !startswith(k,"legacy/") && !(k in ("parameter_checksum","solution_checksum")),exact))

archive_checks = Dict{String,Any}()
try
    restored = load_mmgnlse_solution(joinpath(root,"main_archive.jld2");
        expected_checksum=a["solution_checksum"])
    archive_checks["portable_load"] = true
    archive_checks["portable_fields_equal"] = restored.fields == a["agarwal/scalar_dense/forward/rk4ip"]
    archive_checks["portable_default_shock"] = hasproperty(restored.parameters,:self_steepening) ? restored.parameters.self_steepening : true
    replay = solve_mmgnlse(restored.initial_field, restored.parameters, restored.dz; saveat=:steps)
    archive_checks["portable_replay_equal"] = replay.fields == restored.fields
catch err
    archive_checks["portable_error"] = sprint(showerror,err)
end
try
    raw = JLD2.load(joinpath(root,"main_raw_archive.jld2"))
    archive_checks["raw_forward_type"] = string(typeof(raw["forward"]))
    archive_checks["raw_parameter_type"] = string(typeof(raw["forward"].parameters))
    archive_checks["raw_parameter_canonical"] = raw["forward"].parameters isa PP.MMGNLSEParameters
    archive_checks["raw_fields_equal"] = raw["forward"].fields == a["agarwal/scalar_dense/forward/rk4ip"]
    try
        replay = solve_mmgnlse(raw["forward"].initial_field,raw["forward"].parameters,raw["forward"].dz; saveat=:steps)
        archive_checks["raw_replay_equal"] = replay.fields == raw["forward"].fields
    catch err
        archive_checks["raw_replay_error"] = sprint(showerror,err)
    end
catch err
    archive_checks["raw_load_error"] = sprint(showerror,err)
end
summary["archive"] = archive_checks
for key in sort!(collect(keys(archive_checks)))
    println("ARCHIVE ",key," ",archive_checks[key])
end
open(joinpath(root,"integrated_audit_results.toml"),"w") do io
    TOML.print(io,summary; sorted=true)
end
