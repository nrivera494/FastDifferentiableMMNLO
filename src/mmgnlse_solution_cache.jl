const _MMGNLSE_SOLUTION_KINDS = (
    "MMGNLSESolution",
    "MMGNLSEAdjointSolution",
)

function _mmgnlse_solution_cache_paths(data_path::AbstractString,
                                       metadata_path)
    data = String(data_path)
    metadata = if metadata_path === nothing
        base, extension = splitext(data)
        lowercase(extension) == ".jld2" || throw(ArgumentError(
            "The solution data path must have a .jld2 extension; got $data."))
        base * ".toml"
    elseif metadata_path isa AbstractString
        String(metadata_path)
    else
        throw(ArgumentError(
            "metadata_path must be a .toml path or nothing."))
    end
    return _validate_cache_paths(data, metadata)
end

_mmgnlse_required_payload(data, key::AbstractString) =
    haskey(data, key) ? data[key] : throw(ArgumentError(
        "MMGNLSE solution JLD2 payload is missing required dataset $key."))

_mmgnlse_portable_array(array::Array) = array
_mmgnlse_portable_array(array::AbstractArray) = Array(array)

function _mmgnlse_beta_logical_value(beta::TaylorBeta)
    return (kind=:TaylorBeta, coefficients=beta.coefficients)
end

function _mmgnlse_beta_logical_value(beta::SampledBeta)
    return (kind=:SampledBeta, omega=beta.omega, values=beta.values)
end

function _mmgnlse_overlap_logical_value(overlap::SpatialOverlap)
    return (kind=:SpatialOverlap, values=overlap.values)
end

function _mmgnlse_overlap_logical_value(overlap::ComponentOverlap)
    return (kind=:ComponentOverlap, values=overlap.values)
end

function _mmgnlse_overlap_logical_value(
    overlap::MMGNLSECPDecomposition)
    return (
        kind=:MMGNLSECPDecomposition,
        weights=overlap.λ,
        factors=overlap.U,
        layout=overlap.metadata.layout,
        source_size=overlap.metadata.source_size,
        nmodes=overlap.metadata.nmodes,
        npolarizations=overlap.metadata.npolarizations,
        source_checksum=overlap.metadata.source_checksum,
        relative_error=overlap.relative_error,
        seed=overlap.seed,
        iterations=overlap.iterations,
        converged=overlap.converged,
        zero_tensor=overlap.zero_tensor,
    )
end

_mmgnlse_raman_logical_value(::NoRaman) = (kind=:NoRaman,)
function _mmgnlse_raman_logical_value(response::AgarwalRaman)
    return (
        kind=:AgarwalRaman,
        fraction=response.fraction,
        tau1=response.tau1,
        tau2=response.tau2,
    )
end
function _mmgnlse_raman_logical_value(response::AnisotropicRaman)
    return (
        kind=:AnisotropicRaman,
        fraction=response.fraction,
        fa=response.fa,
        fb=response.fb,
        fc=response.fc,
        tau1=response.tau1,
        tau2=response.tau2,
        taub=response.taub,
    )
end

function _mmgnlse_parameters_logical_value(parameters::MMGNLSEParameters)
    grid = parameters.domain.time_grid
    grid_value = grid === nothing ? nothing : (nt=grid.nt, dt=grid.dt)
    return (
        domain=(dofs=parameters.domain.dofs.names, time_grid=grid_value),
        length=parameters.length,
        alpha=parameters.alpha,
        gain=parameters.gain,
        beta=_mmgnlse_beta_logical_value(parameters.beta),
        overlap=_mmgnlse_overlap_logical_value(parameters.S),
        n2=parameters.n2,
        omega0=parameters.omega0,
        raman=_mmgnlse_raman_logical_value(parameters.raman),
    )
end

"""
    mmgnlse_parameters_checksum(parameters) -> String

Return a deterministic SHA-256 checksum of every physical and discretization
value in an `MMGNLSEParameters` object.
"""
mmgnlse_parameters_checksum(parameters::MMGNLSEParameters) =
    _canonical_value_checksum(_mmgnlse_parameters_logical_value(parameters))

function _mmgnlse_parameter_payload(parameters::MMGNLSEParameters)
    payload = Dict{String,Any}(
        "parameter_dofs" => String.(parameters.domain.dofs.names),
        "parameter_has_time_grid" => parameters.domain.time_grid !== nothing,
        "parameter_length" => parameters.length,
        "parameter_alpha" => _mmgnlse_portable_array(parameters.alpha),
        "parameter_gain" => _mmgnlse_portable_array(parameters.gain),
        "parameter_n2" => parameters.n2,
        "parameter_omega0" => parameters.omega0,
    )
    if parameters.domain.time_grid !== nothing
        payload["parameter_time_nt"] = parameters.domain.time_grid.nt
        payload["parameter_time_dt"] = parameters.domain.time_grid.dt
    end

    beta = parameters.beta
    if beta isa TaylorBeta
        payload["parameter_beta_kind"] = "TaylorBeta"
        payload["parameter_beta_coefficients"] =
            _mmgnlse_portable_array(beta.coefficients)
    elseif beta isa SampledBeta
        payload["parameter_beta_kind"] = "SampledBeta"
        payload["parameter_beta_omega"] =
            _mmgnlse_portable_array(beta.omega)
        payload["parameter_beta_values"] =
            _mmgnlse_portable_array(beta.values)
    else
        throw(ArgumentError(
            "Unsupported MMGNLSE beta type $(typeof(beta))."))
    end

    overlap = parameters.S
    if overlap isa SpatialOverlap
        payload["parameter_overlap_kind"] = "SpatialOverlap"
        payload["parameter_overlap_values"] =
            _mmgnlse_portable_array(overlap.values)
    elseif overlap isa ComponentOverlap
        payload["parameter_overlap_kind"] = "ComponentOverlap"
        payload["parameter_overlap_values"] =
            _mmgnlse_portable_array(overlap.values)
    elseif overlap isa MMGNLSECPDecomposition
        payload["parameter_overlap_kind"] = "MMGNLSECPDecomposition"
        payload["parameter_cp_weights"] =
            _mmgnlse_portable_array(overlap.λ)
        for index in 1:4
            payload["parameter_cp_factor_$index"] =
                _mmgnlse_portable_array(overlap.U[index])
        end
        payload["parameter_cp_layout"] = String(overlap.metadata.layout)
        payload["parameter_cp_nmodes"] = overlap.metadata.nmodes
        payload["parameter_cp_npolarizations"] =
            overlap.metadata.npolarizations
        payload["parameter_cp_has_source_checksum"] =
            overlap.metadata.source_checksum !== nothing
        payload["parameter_cp_source_checksum"] =
            something(overlap.metadata.source_checksum, "")
        payload["parameter_cp_relative_error"] = overlap.relative_error
        payload["parameter_cp_seed"] = overlap.seed
        payload["parameter_cp_iterations"] = overlap.iterations
        payload["parameter_cp_converged"] = overlap.converged
        payload["parameter_cp_zero_tensor"] = overlap.zero_tensor
    else
        throw(ArgumentError(
            "Unsupported MMGNLSE overlap type $(typeof(overlap))."))
    end

    response = parameters.raman
    if response isa NoRaman
        payload["parameter_raman_kind"] = "NoRaman"
    elseif response isa AgarwalRaman
        payload["parameter_raman_kind"] = "AgarwalRaman"
        payload["parameter_raman_fraction"] = response.fraction
        payload["parameter_raman_tau1"] = response.tau1
        payload["parameter_raman_tau2"] = response.tau2
    elseif response isa AnisotropicRaman
        payload["parameter_raman_kind"] = "AnisotropicRaman"
        payload["parameter_raman_fraction"] = response.fraction
        payload["parameter_raman_fa"] = response.fa
        payload["parameter_raman_fb"] = response.fb
        payload["parameter_raman_fc"] = response.fc
        payload["parameter_raman_tau1"] = response.tau1
        payload["parameter_raman_tau2"] = response.tau2
        payload["parameter_raman_taub"] = response.taub
    else
        throw(ArgumentError(
            "Unsupported MMGNLSE Raman type $(typeof(response))."))
    end
    return payload
end

function _mmgnlse_load_beta(data)
    kind = String(_mmgnlse_required_payload(data, "parameter_beta_kind"))
    if kind == "TaylorBeta"
        return TaylorBeta(_mmgnlse_required_payload(
            data, "parameter_beta_coefficients"))
    elseif kind == "SampledBeta"
        return SampledBeta(
            _mmgnlse_required_payload(data, "parameter_beta_omega"),
            _mmgnlse_required_payload(data, "parameter_beta_values"))
    end
    throw(ArgumentError("Unsupported cached MMGNLSE beta kind $kind."))
end

function _mmgnlse_load_overlap(data)
    kind = String(_mmgnlse_required_payload(
        data, "parameter_overlap_kind"))
    if kind == "SpatialOverlap"
        return SpatialOverlap(_mmgnlse_required_payload(
            data, "parameter_overlap_values"))
    elseif kind == "ComponentOverlap"
        return ComponentOverlap(_mmgnlse_required_payload(
            data, "parameter_overlap_values"))
    elseif kind == "MMGNLSECPDecomposition"
        factors = ntuple(index -> _mmgnlse_required_payload(
            data, "parameter_cp_factor_$index"), 4)
        has_source = Bool(_mmgnlse_required_payload(
            data, "parameter_cp_has_source_checksum"))
        source_checksum = has_source ? String(_mmgnlse_required_payload(
            data, "parameter_cp_source_checksum")) : nothing
        return MMGNLSECPDecomposition(
            _mmgnlse_required_payload(data, "parameter_cp_weights"),
            factors;
            layout=Symbol(_mmgnlse_required_payload(
                data, "parameter_cp_layout")),
            nmodes=Int(_mmgnlse_required_payload(
                data, "parameter_cp_nmodes")),
            npolarizations=Int(_mmgnlse_required_payload(
                data, "parameter_cp_npolarizations")),
            source_checksum,
            relative_error=_mmgnlse_required_payload(
                data, "parameter_cp_relative_error"),
            seed=Int(_mmgnlse_required_payload(
                data, "parameter_cp_seed")),
            iterations=Int(_mmgnlse_required_payload(
                data, "parameter_cp_iterations")),
            converged=Bool(_mmgnlse_required_payload(
                data, "parameter_cp_converged")),
            zero_tensor=Bool(_mmgnlse_required_payload(
                data, "parameter_cp_zero_tensor")))
    end
    throw(ArgumentError("Unsupported cached MMGNLSE overlap kind $kind."))
end

function _mmgnlse_load_raman(data)
    kind = String(_mmgnlse_required_payload(
        data, "parameter_raman_kind"))
    if kind == "NoRaman"
        return NoRaman()
    elseif kind == "AgarwalRaman"
        return AgarwalRaman(
            fraction=_mmgnlse_required_payload(
                data, "parameter_raman_fraction"),
            tau1=_mmgnlse_required_payload(data, "parameter_raman_tau1"),
            tau2=_mmgnlse_required_payload(data, "parameter_raman_tau2"))
    elseif kind == "AnisotropicRaman"
        return AnisotropicRaman(
            fraction=_mmgnlse_required_payload(
                data, "parameter_raman_fraction"),
            fa=_mmgnlse_required_payload(data, "parameter_raman_fa"),
            fb=_mmgnlse_required_payload(data, "parameter_raman_fb"),
            fc=_mmgnlse_required_payload(data, "parameter_raman_fc"),
            tau1=_mmgnlse_required_payload(data, "parameter_raman_tau1"),
            tau2=_mmgnlse_required_payload(data, "parameter_raman_tau2"),
            taub=_mmgnlse_required_payload(data, "parameter_raman_taub"))
    end
    throw(ArgumentError("Unsupported cached MMGNLSE Raman kind $kind."))
end

function _mmgnlse_load_parameters(data)
    names = Tuple(Symbol.(String.(_mmgnlse_required_payload(
        data, "parameter_dofs"))))
    has_grid = Bool(_mmgnlse_required_payload(
        data, "parameter_has_time_grid"))
    grid = has_grid ? TimeGrid{typeof(_mmgnlse_required_payload(
        data, "parameter_time_dt"))}(
            Int(_mmgnlse_required_payload(data, "parameter_time_nt")),
            _mmgnlse_required_payload(data, "parameter_time_dt")) : nothing
    domain = MMGNLSEDomain(
        degrees_of_freedom(names...); time_grid=grid)
    return MMGNLSEParameters(
        domain;
        length=_mmgnlse_required_payload(data, "parameter_length"),
        alpha=_mmgnlse_required_payload(data, "parameter_alpha"),
        gain=_mmgnlse_required_payload(data, "parameter_gain"),
        beta=_mmgnlse_load_beta(data),
        S=_mmgnlse_load_overlap(data),
        n2=_mmgnlse_required_payload(data, "parameter_n2"),
        omega0=_mmgnlse_required_payload(data, "parameter_omega0"),
        raman=_mmgnlse_load_raman(data))
end

function _mmgnlse_forward_integration_z(solution::MMGNLSESolution)
    integration_z = _mmgnlse_cache_integration_z(solution)
    return integration_z === nothing ? Float64[] :
           Float64.(Array(integration_z))
end

function _mmgnlse_method_name(method)
    name = String(nameof(typeof(method)))
    name in ("RK4IP", "Tsit5") || throw(ArgumentError(
        "Cannot archive unsupported MMGNLSE forward method $(typeof(method))."))
    return name
end

function _mmgnlse_load_method(name)
    method = String(name)
    method == "RK4IP" && return RK4IP()
    method == "Tsit5" && return Tsit5()
    throw(ArgumentError("Unsupported cached MMGNLSE method $method."))
end

function _mmgnlse_solution_logical_value(
    solution::MMGNLSESolution,
    parameters_checksum::AbstractString=mmgnlse_parameters_checksum(
        solution.parameters))
    return (
        kind=:MMGNLSESolution,
        z=_mmgnlse_portable_array(solution.z),
        fields=_mmgnlse_portable_array(solution.fields),
        parameters_checksum=String(parameters_checksum),
        initial_field=_mmgnlse_portable_array(solution.initial_field),
        dz=solution.dz,
        method=Symbol(_mmgnlse_method_name(solution.method)),
        integration_z=_mmgnlse_forward_integration_z(solution),
    )
end

function _mmgnlse_solution_logical_value(
    solution::MMGNLSEAdjointSolution,
    parameters_checksum::AbstractString=mmgnlse_parameters_checksum(
        solution.parameters))
    return (
        kind=:MMGNLSEAdjointSolution,
        z=_mmgnlse_portable_array(solution.z),
        fields=_mmgnlse_portable_array(solution.fields),
        parameters_checksum=String(parameters_checksum),
        units=solution.units,
    )
end

"""
    mmgnlse_solution_checksum(solution) -> String

Return a deterministic SHA-256 checksum of the portable public content of a
forward or adjoint MMGNLSE solution. Runtime solver caches are excluded.
"""
mmgnlse_solution_checksum(
    solution::Union{MMGNLSESolution,MMGNLSEAdjointSolution}) =
    _canonical_value_checksum(_mmgnlse_solution_logical_value(solution))

function _mmgnlse_validate_solution_archive(
    solution::MMGNLSESolution)
    parameters = solution.parameters
    validate_public_field(
        solution.fields, parameters.domain,
        parameter_mode_count(parameters); nz=length(solution.z),
        name=:solution_fields)
    validate_public_field(
        solution.initial_field, parameters.domain,
        parameter_mode_count(parameters), name=:solution_initial_field)
    length(solution.z) > 0 && all(isfinite, solution.z) ||
        throw(ArgumentError("Forward solution z values must be finite."))
    all(diff(solution.z) .> 0) || throw(ArgumentError(
        "Forward solution z values must be strictly increasing."))
    first(solution.z) == 0 && last(solution.z) == parameters.length ||
        throw(ArgumentError(
            "Forward solution z must span 0 to parameters.length."))
    isfinite(solution.dz) && solution.dz > 0 || throw(ArgumentError(
        "Forward solution dz must be finite and positive."))
    all(_mmgnlse_isfinite, solution.fields) || throw(ArgumentError(
        "Forward solution fields must be finite."))
    all(_mmgnlse_isfinite, solution.initial_field) ||
        throw(ArgumentError("Forward solution initial_field must be finite."))
    _mmgnlse_method_name(solution.method)
    integration_z = _mmgnlse_forward_integration_z(solution)
    if !isempty(integration_z)
        first(integration_z) == 0 &&
            last(integration_z) == parameters.length ||
            throw(ArgumentError(
                "Forward integration_z must span 0 to parameters.length."))
        all(diff(integration_z) .> 0) || throw(ArgumentError(
            "Forward integration_z must be strictly increasing."))
    end
    return solution
end

function _mmgnlse_validate_solution_archive(
    solution::MMGNLSEAdjointSolution)
    parameters = solution.parameters
    validate_public_field(
        solution.fields, parameters.domain,
        parameter_mode_count(parameters); nz=length(solution.z),
        name=:adjoint_fields)
    length(solution.z) > 0 && all(isfinite, solution.z) ||
        throw(ArgumentError("Adjoint solution z values must be finite."))
    all(diff(solution.z) .> 0) || throw(ArgumentError(
        "Adjoint solution z values must be strictly increasing."))
    first(solution.z) == 0 && last(solution.z) == parameters.length ||
        throw(ArgumentError(
            "Adjoint solution z must span 0 to parameters.length."))
    solution.units in (:power, :photon) || throw(ArgumentError(
        "Adjoint solution units must be :power or :photon."))
    all(_mmgnlse_isfinite, solution.fields) || throw(ArgumentError(
        "Adjoint solution fields must be finite."))
    return solution
end

function _mmgnlse_solution_payload(
    solution::Union{MMGNLSESolution,MMGNLSEAdjointSolution})
    parameter_payload = _mmgnlse_parameter_payload(solution.parameters)
    payload = Dict{String,Any}(
        "solution_kind" => String(nameof(typeof(solution))),
        "solution_z" => _mmgnlse_portable_array(solution.z),
        "solution_fields" => _mmgnlse_portable_array(solution.fields),
    )
    merge!(payload, parameter_payload)
    if solution isa MMGNLSESolution
        payload["solution_initial_field"] =
            _mmgnlse_portable_array(solution.initial_field)
        payload["solution_dz"] = solution.dz
        payload["solution_method"] = _mmgnlse_method_name(solution.method)
        payload["solution_integration_z"] =
            _mmgnlse_forward_integration_z(solution)
    else
        payload["solution_units"] = String(solution.units)
    end
    return payload
end

function _mmgnlse_payload_array_checksums(payload)
    checksums = Dict{String,Any}()
    for (name, value) in payload
        value isa AbstractArray{<:Number} || continue
        checksums[name] = canonical_array_checksum(value)
    end
    return checksums
end

function _mmgnlse_write_solution_payload(path, payload)
    JLD2.jldopen(path, "w") do file
        for name in sort!(collect(keys(payload)))
            file[name] = payload[name]
        end
    end
    return nothing
end

function _mmgnlse_solution_backend(solution)
    cache = solution.cache
    cache === nothing && return "unknown"
    hasproperty(cache, :backend) || return "cpu"
    return String(getproperty(cache, :backend))
end

"""
    save_mmgnlse_solution(data_path, solution;
                          metadata_path=nothing, overwrite=false)

Save a forward or adjoint MMGNLSE solution as a versioned JLD2 payload and
checksum-validated TOML sidecar. By default, `metadata_path` is obtained by
replacing the `.jld2` extension with `.toml`. Complete parameters are embedded;
runtime solver caches and ODE objects are not serialized.
"""
function save_mmgnlse_solution(
    data_path::AbstractString,
    solution::Union{MMGNLSESolution,MMGNLSEAdjointSolution};
    metadata_path=nothing,
    overwrite::Bool=false)
    _mmgnlse_validate_solution_archive(solution)
    data, metadata_path_value = _mmgnlse_solution_cache_paths(
        data_path, metadata_path)
    payload = _mmgnlse_solution_payload(solution)
    parameters_checksum = mmgnlse_parameters_checksum(solution.parameters)
    payload_checksum = _canonical_value_checksum(
        _mmgnlse_solution_logical_value(solution, parameters_checksum))
    kind = String(nameof(typeof(solution)))
    provenance = Dict{String,Any}(
        "solution_kind" => kind,
        "field_shape" => collect(size(solution.fields)),
        "saved_planes" => length(solution.z),
        "parameters_checksum" => parameters_checksum,
        "backend" => _mmgnlse_solution_backend(solution),
    )
    if solution isa MMGNLSESolution
        provenance["method"] = _mmgnlse_method_name(solution.method)
        provenance["dz"] = Float64(solution.dz)
        provenance["integration_steps"] =
            length(_mmgnlse_forward_integration_z(solution))
        provenance["stored_units"] = "power"
    else
        provenance["stored_units"] = String(solution.units)
    end
    metadata = Dict{String,Any}(
        "format" => _MMGNLSE_CACHE_FORMAT,
        "schema_version" => MMGNLSE_CACHE_SCHEMA_VERSION,
        "checksum_algorithm" => _MMGNLSE_CACHE_CHECKSUM_ALGORITHM,
        "canonical_value_format" => _CANONICAL_VALUE_FORMAT,
        "kind" => kind,
        "payload_checksum" => payload_checksum,
        "parameters_checksum" => parameters_checksum,
        "arrays" => _mmgnlse_payload_array_checksums(payload),
        "provenance" => provenance,
    )
    writer = path -> _mmgnlse_write_solution_payload(path, payload)
    return _write_cache_pair(
        data, metadata_path_value, metadata, writer; overwrite)
end

function _mmgnlse_read_solution_metadata(data_path, metadata_path)
    isfile(metadata_path) || throw(ArgumentError(
        "Cache metadata file does not exist: $metadata_path."))
    preview = TOML.parsefile(metadata_path)
    kind = get(preview, "kind", nothing)
    kind in _MMGNLSE_SOLUTION_KINDS || throw(ArgumentError(
        "Unsupported MMGNLSE solution kind $(repr(kind)) in $metadata_path."))
    return _read_cache_metadata(data_path, metadata_path, kind)
end

function _mmgnlse_validate_payload_arrays(data, metadata)
    arrays = get(metadata, "arrays", nothing)
    arrays isa AbstractDict || throw(ArgumentError(
        "MMGNLSE solution metadata is missing array checksums."))
    for (name, expected) in arrays
        value = _mmgnlse_required_payload(data, String(name))
        value isa AbstractArray{<:Number} || throw(ArgumentError(
            "Cached dataset $name is not a numeric array."))
        _require_checksum_match(
            canonical_array_checksum(value), expected,
            "MMGNLSE solution $name")
    end
    return nothing
end

function _mmgnlse_select_loaded_parameters(archived, supplied, metadata)
    archived_checksum = mmgnlse_parameters_checksum(archived)
    _require_checksum_match(
        archived_checksum, get(metadata, "parameters_checksum", nothing),
        "archived MMGNLSE parameters")
    supplied === nothing && return archived, archived_checksum
    supplied isa MMGNLSEParameters || throw(ArgumentError(
        "parameters must be an MMGNLSEParameters object or nothing."))
    supplied_checksum = mmgnlse_parameters_checksum(supplied)
    _require_checksum_match(
        supplied_checksum, archived_checksum,
        "supplied MMGNLSE parameters")
    return supplied, archived_checksum
end

function _mmgnlse_load_forward_solution(data, parameters)
    z = _mmgnlse_required_payload(data, "solution_z")
    fields = _mmgnlse_required_payload(data, "solution_fields")
    initial_field = _mmgnlse_required_payload(
        data, "solution_initial_field")
    dz = _mmgnlse_required_payload(data, "solution_dz")
    method = _mmgnlse_load_method(_mmgnlse_required_payload(
        data, "solution_method"))
    integration_z = _mmgnlse_required_payload(
        data, "solution_integration_z")
    cache = isempty(integration_z) ? nothing :
            (; integration_z, restored=true)
    return MMGNLSESolution(
        z=z, fields=fields, parameters=parameters,
        initial_field=initial_field, dz=dz, method=method, cache=cache)
end

function _mmgnlse_load_adjoint_solution(data, parameters)
    units = Symbol(_mmgnlse_required_payload(data, "solution_units"))
    units in (:power, :photon) || throw(ArgumentError(
        "Cached adjoint units must be :power or :photon."))
    return MMGNLSEAdjointSolution(
        z=_mmgnlse_required_payload(data, "solution_z"),
        fields=_mmgnlse_required_payload(data, "solution_fields"),
        parameters=parameters,
        units=units,
        cache=nothing)
end

"""
    load_mmgnlse_solution(data_path;
                          metadata_path=nothing,
                          parameters=nothing,
                          expected_checksum=nothing)

Load and validate a forward or adjoint MMGNLSE solution. If `parameters` is
omitted, the embedded parameters are reconstructed. If supplied, their
deterministic checksum must match the archive and that exact object is attached
to the returned solution.
"""
function load_mmgnlse_solution(
    data_path::AbstractString;
    metadata_path=nothing,
    parameters=nothing,
    expected_checksum=nothing)
    data, metadata_path_value = _mmgnlse_solution_cache_paths(
        data_path, metadata_path)
    metadata = _mmgnlse_read_solution_metadata(
        data, metadata_path_value)
    payload = JLD2.load(data)
    _mmgnlse_validate_payload_arrays(payload, metadata)
    archived_parameters = _mmgnlse_load_parameters(payload)
    attached_parameters, parameters_checksum =
        _mmgnlse_select_loaded_parameters(
            archived_parameters, parameters, metadata)
    kind = String(_mmgnlse_required_payload(
        payload, "solution_kind"))
    get(metadata, "kind", nothing) == kind || throw(ArgumentError(
        "MMGNLSE solution kind differs between JLD2 and TOML."))
    solution = if kind == "MMGNLSESolution"
        _mmgnlse_load_forward_solution(payload, attached_parameters)
    elseif kind == "MMGNLSEAdjointSolution"
        _mmgnlse_load_adjoint_solution(payload, attached_parameters)
    else
        throw(ArgumentError("Unsupported cached MMGNLSE solution kind $kind."))
    end
    _mmgnlse_validate_solution_archive(solution)
    payload_checksum = _canonical_value_checksum(
        _mmgnlse_solution_logical_value(solution, parameters_checksum))
    _require_checksum_match(
        payload_checksum, get(metadata, "payload_checksum", nothing),
        "MMGNLSE solution payload")
    if expected_checksum !== nothing
        _require_checksum_match(
            payload_checksum, expected_checksum,
            "expected MMGNLSE solution payload")
    end
    return solution
end
