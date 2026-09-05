import JLD2
import SHA
import TOML

const MMGNLSE_CACHE_SCHEMA_VERSION = 1
const _MMGNLSE_CACHE_FORMAT = "PulsePropagation-MMGNLSE-cache"
const _CANONICAL_VALUE_FORMAT = "PulsePropagation-canonical-value-v1"
const _MMGNLSE_CACHE_CHECKSUM_ALGORITHM = "SHA-256"

"""
    canonical_array_checksum(array) -> String

Return a canonical SHA-256 checksum for a dense numeric array. The digest
contains the element type, dimensionality, shape, and the exact element bits
in Julia column-major order. Integer and floating-point bits are encoded in a
fixed little-endian representation, so the checksum does not depend on host
endianness or Julia serialization details.

The function intentionally rejects unsupported numeric representations rather
than falling back to a potentially unstable textual or serialized form.
"""
function canonical_array_checksum(array::AbstractArray{<:Number})
    context = SHA.SHA2_256_CTX()
    _canonical_update_token!(context, _CANONICAL_VALUE_FORMAT)
    _canonical_update_token!(context, "numeric-array")
    _canonical_update_token!(context, _canonical_number_tag(eltype(array)))
    _canonical_update_u64!(context, UInt64(ndims(array)))
    for dimension in size(array)
        _canonical_update_u64!(context, UInt64(dimension))
    end
    for value in array
        _canonical_update_number_bits!(context, value)
    end
    return bytes2hex(SHA.digest!(context))
end

function canonical_array_checksum(array::AbstractArray)
    throw(ArgumentError(
        "canonical_array_checksum supports dense numeric arrays; got eltype $(eltype(array))."))
end

_canonical_number_tag(::Type{Bool}) = "Bool"
_canonical_number_tag(::Type{Int8}) = "Int8"
_canonical_number_tag(::Type{Int16}) = "Int16"
_canonical_number_tag(::Type{Int32}) = "Int32"
_canonical_number_tag(::Type{Int64}) = "Int64"
_canonical_number_tag(::Type{Int128}) = "Int128"
_canonical_number_tag(::Type{UInt8}) = "UInt8"
_canonical_number_tag(::Type{UInt16}) = "UInt16"
_canonical_number_tag(::Type{UInt32}) = "UInt32"
_canonical_number_tag(::Type{UInt64}) = "UInt64"
_canonical_number_tag(::Type{UInt128}) = "UInt128"
_canonical_number_tag(::Type{Float16}) = "Float16"
_canonical_number_tag(::Type{Float32}) = "Float32"
_canonical_number_tag(::Type{Float64}) = "Float64"
_canonical_number_tag(::Type{Complex{T}}) where {T} =
    "Complex{" * _canonical_number_tag(T) * "}"
function _canonical_number_tag(::Type{T}) where {T<:Number}
    throw(ArgumentError(
        "Canonical checksums do not support numeric element type $T."))
end

function _canonical_update_token!(context, token::AbstractString)
    bytes = collect(codeunits(token))
    _canonical_update_u64!(context, UInt64(length(bytes)))
    SHA.update!(context, bytes)
    return context
end

function _canonical_update_unsigned!(context, value::Unsigned, nbytes::Int)
    bytes = ntuple(index -> UInt8((value >> (8 * (index - 1))) & 0xff), nbytes)
    SHA.update!(context, bytes)
    return context
end

_canonical_update_u64!(context, value::UInt64) =
    _canonical_update_unsigned!(context, value, 8)

_canonical_update_number_bits!(context, value::Bool) =
    _canonical_update_unsigned!(context, UInt8(value), 1)
_canonical_update_number_bits!(context, value::Int8) =
    _canonical_update_unsigned!(context, reinterpret(UInt8, value), 1)
_canonical_update_number_bits!(context, value::Int16) =
    _canonical_update_unsigned!(context, reinterpret(UInt16, value), 2)
_canonical_update_number_bits!(context, value::Int32) =
    _canonical_update_unsigned!(context, reinterpret(UInt32, value), 4)
_canonical_update_number_bits!(context, value::Int64) =
    _canonical_update_unsigned!(context, reinterpret(UInt64, value), 8)
_canonical_update_number_bits!(context, value::Int128) =
    _canonical_update_unsigned!(context, reinterpret(UInt128, value), 16)
_canonical_update_number_bits!(context, value::UInt8) =
    _canonical_update_unsigned!(context, value, 1)
_canonical_update_number_bits!(context, value::UInt16) =
    _canonical_update_unsigned!(context, value, 2)
_canonical_update_number_bits!(context, value::UInt32) =
    _canonical_update_unsigned!(context, value, 4)
_canonical_update_number_bits!(context, value::UInt64) =
    _canonical_update_unsigned!(context, value, 8)
_canonical_update_number_bits!(context, value::UInt128) =
    _canonical_update_unsigned!(context, value, 16)
_canonical_update_number_bits!(context, value::Float16) =
    _canonical_update_unsigned!(context, reinterpret(UInt16, value), 2)
_canonical_update_number_bits!(context, value::Float32) =
    _canonical_update_unsigned!(context, reinterpret(UInt32, value), 4)
_canonical_update_number_bits!(context, value::Float64) =
    _canonical_update_unsigned!(context, reinterpret(UInt64, value), 8)
function _canonical_update_number_bits!(context, value::Complex)
    _canonical_update_number_bits!(context, real(value))
    _canonical_update_number_bits!(context, imag(value))
    return context
end
function _canonical_update_number_bits!(context, value::Number)
    _canonical_number_tag(typeof(value))
    error("Internal checksum encoder is missing an implementation for $(typeof(value)).")
end

# Recursive logical-value hashing is used for cache provenance. Numeric arrays
# use the same representation as `canonical_array_checksum`; heterogeneous
# arrays and structs are encoded field by field. Unsupported objects (notably
# closures) fail explicitly instead of generating session-dependent hashes.
function _canonical_value_checksum(value)
    context = SHA.SHA2_256_CTX()
    _canonical_update_token!(context, _CANONICAL_VALUE_FORMAT)
    _canonical_update_value!(context, value)
    return bytes2hex(SHA.digest!(context))
end

function _canonical_update_value!(context, value::AbstractArray{<:Number})
    _canonical_update_token!(context, "numeric-array")
    _canonical_update_token!(context, _canonical_number_tag(eltype(value)))
    _canonical_update_u64!(context, UInt64(ndims(value)))
    for dimension in size(value)
        _canonical_update_u64!(context, UInt64(dimension))
    end
    for entry in value
        _canonical_update_number_bits!(context, entry)
    end
    return context
end

function _canonical_update_value!(context, value::AbstractArray)
    _canonical_update_token!(context, "array")
    _canonical_update_u64!(context, UInt64(ndims(value)))
    for dimension in size(value)
        _canonical_update_u64!(context, UInt64(dimension))
    end
    for entry in value
        _canonical_update_value!(context, entry)
    end
    return context
end

function _canonical_update_value!(context, value::Number)
    _canonical_update_token!(context, "number")
    _canonical_update_token!(context, _canonical_number_tag(typeof(value)))
    _canonical_update_number_bits!(context, value)
    return context
end

function _canonical_update_value!(context, value::AbstractString)
    _canonical_update_token!(context, "string")
    _canonical_update_token!(context, value)
    return context
end

function _canonical_update_value!(context, value::Symbol)
    _canonical_update_token!(context, "symbol")
    _canonical_update_token!(context, String(value))
    return context
end

function _canonical_update_value!(context, ::Nothing)
    _canonical_update_token!(context, "nothing")
    return context
end

function _canonical_update_value!(context, value::NamedTuple)
    _canonical_update_token!(context, "named-tuple")
    _canonical_update_u64!(context, UInt64(length(value)))
    for name in keys(value)
        _canonical_update_token!(context, String(name))
        _canonical_update_value!(context, getproperty(value, name))
    end
    return context
end

function _canonical_update_value!(context, value::Tuple)
    _canonical_update_token!(context, "tuple")
    _canonical_update_u64!(context, UInt64(length(value)))
    for entry in value
        _canonical_update_value!(context, entry)
    end
    return context
end

function _canonical_update_value!(context, value)
    type = typeof(value)
    isstructtype(type) && !(value isa Function) || throw(ArgumentError(
        "Cannot generate deterministic cache provenance for $(typeof(value))."))
    _canonical_update_token!(context, "struct")
    _canonical_update_token!(context, String(nameof(type)))
    names = fieldnames(type)
    _canonical_update_u64!(context, UInt64(length(names)))
    for (index, name) in enumerate(names)
        _canonical_update_token!(context, String(name))
        _canonical_update_value!(context, getfield(value, index))
    end
    return context
end

function _file_sha256(path::AbstractString)
    return open(path, "r") do input
        bytes2hex(SHA.sha256(input))
    end
end

function _validate_cache_paths(data_path::AbstractString,
                               metadata_path::AbstractString)
    isempty(data_path) && throw(ArgumentError("The JLD2 data path cannot be empty."))
    isempty(metadata_path) && throw(ArgumentError("The TOML metadata path cannot be empty."))
    abspath(data_path) == abspath(metadata_path) && throw(ArgumentError(
        "The JLD2 data path and TOML metadata path must be different."))
    lowercase(splitext(data_path)[2]) == ".jld2" || throw(ArgumentError(
        "The cache data path must have a .jld2 extension; got $data_path."))
    lowercase(splitext(metadata_path)[2]) == ".toml" || throw(ArgumentError(
        "The cache metadata path must have a .toml extension; got $metadata_path."))
    return (String(data_path), String(metadata_path))
end

function _prepare_cache_destination(data_path::AbstractString,
                                    metadata_path::AbstractString,
                                    overwrite::Bool)
    data_path, metadata_path = _validate_cache_paths(data_path, metadata_path)
    existing = filter(ispath, (data_path, metadata_path))
    isempty(existing) || overwrite || throw(ArgumentError(
        "Cache path already exists ($(join(existing, ", "))); pass overwrite=true to replace it."))
    for path in (data_path, metadata_path)
        parent = dirname(abspath(path))
        isdir(parent) || throw(ArgumentError(
            "Cache parent directory does not exist: $parent."))
    end
    return (data_path, metadata_path)
end

function _write_cache_pair(data_path::AbstractString,
                           metadata_path::AbstractString,
                           metadata::AbstractDict,
                           write_data!;
                           overwrite::Bool=false)
    data_path, metadata_path = _prepare_cache_destination(
        data_path, metadata_path, overwrite)
    temporary_data = tempname(dirname(abspath(data_path))) * ".jld2"
    temporary_metadata = tempname(dirname(abspath(metadata_path))) * ".toml"
    try
        write_data!(temporary_data)
        complete_metadata = Dict{String,Any}(String(key) => value for (key, value) in metadata)
        complete_metadata["jld2_sha256"] = _file_sha256(temporary_data)
        open(temporary_metadata, "w") do output
            TOML.print(output, complete_metadata; sorted=true)
        end
        mv(temporary_data, data_path; force=overwrite)
        mv(temporary_metadata, metadata_path; force=overwrite)
    finally
        ispath(temporary_data) && rm(temporary_data; force=true)
        ispath(temporary_metadata) && rm(temporary_metadata; force=true)
    end
    return (data_path=data_path, metadata_path=metadata_path)
end

function _read_cache_metadata(data_path::AbstractString,
                              metadata_path::AbstractString,
                              kind::AbstractString)
    data_path, metadata_path = _validate_cache_paths(data_path, metadata_path)
    isfile(data_path) || throw(ArgumentError("Cache data file does not exist: $data_path."))
    isfile(metadata_path) || throw(ArgumentError(
        "Cache metadata file does not exist: $metadata_path."))
    metadata = TOML.parsefile(metadata_path)
    get(metadata, "format", nothing) == _MMGNLSE_CACHE_FORMAT || throw(ArgumentError(
        "Unsupported cache format in $metadata_path."))
    get(metadata, "schema_version", nothing) == MMGNLSE_CACHE_SCHEMA_VERSION ||
        throw(ArgumentError("Unsupported cache schema version in $metadata_path."))
    get(metadata, "checksum_algorithm", nothing) == _MMGNLSE_CACHE_CHECKSUM_ALGORITHM ||
        throw(ArgumentError("Unsupported cache checksum algorithm in $metadata_path."))
    get(metadata, "canonical_value_format", nothing) == _CANONICAL_VALUE_FORMAT ||
        throw(ArgumentError("Unsupported canonical value format in $metadata_path."))
    get(metadata, "kind", nothing) == kind || throw(ArgumentError(
        "Cache kind mismatch: expected $kind, found $(get(metadata, "kind", nothing))."))
    expected_file_checksum = get(metadata, "jld2_sha256", nothing)
    expected_file_checksum isa AbstractString || throw(ArgumentError(
        "Cache metadata is missing jld2_sha256."))
    actual_file_checksum = _file_sha256(data_path)
    actual_file_checksum == expected_file_checksum || throw(ArgumentError(
        "JLD2 cache checksum mismatch for $data_path."))
    return metadata
end

function _require_checksum_match(actual::AbstractString,
                                 expected,
                                 description::AbstractString)
    expected isa AbstractString || throw(ArgumentError(
        "Cache metadata is missing $description checksum."))
    actual == expected || throw(ArgumentError("$description checksum mismatch."))
    return actual
end

function _require_sha256(checksum, description::AbstractString)
    checksum isa AbstractString || throw(ArgumentError(
        "$description must be a canonical SHA-256 checksum."))
    length(checksum) == 64 && all(character -> character in "0123456789abcdef", checksum) ||
        throw(ArgumentError(
            "$description must be a 64-character lowercase hexadecimal SHA-256 checksum."))
    return String(checksum)
end

_fiber_overlap_logical_value(overlap::SpatialOverlap) = overlap.values
_fiber_overlap_logical_value(overlap::MMGNLSECPDecomposition) =
    (kind=:MMGNLSECPDecomposition, value=_cp_logical_value(overlap))

function _fiber_logical_value(properties::FiberProperties)
    modes = properties.modes
    return (
        beta_coefficients=properties.beta.coefficients,
        S=_fiber_overlap_logical_value(properties.S),
        raman=properties.raman,
        mode_x=modes.x,
        mode_y=modes.y,
        mode_fields=modes.fields,
        mode_beta0=modes.beta0,
        mode_labels=modes.labels,
        mode_wavelength=modes.wavelength,
        config=properties.config,
        n2=properties.n2,
        omega0=properties.omega0,
    )
end

"""Return the deterministic logical checksum stored for `FiberProperties`."""
fiber_properties_checksum(properties::FiberProperties) =
    _canonical_value_checksum(_fiber_logical_value(properties))

function _fiber_array_checksums(properties::FiberProperties)
    modes = properties.modes
    checksums = Dict{String,Any}(
        "beta" => canonical_array_checksum(properties.beta.coefficients),
        "mode_x" => canonical_array_checksum(modes.x),
        "mode_y" => canonical_array_checksum(modes.y),
        "mode_fields" => canonical_array_checksum(modes.fields),
        "mode_beta0" => canonical_array_checksum(modes.beta0),
    )
    if properties.S isa SpatialOverlap
        checksums["S"] = canonical_array_checksum(properties.S.values)
    else
        for (name, checksum) in _cp_array_checksums(properties.S)
            checksums["S_$name"] = checksum
        end
    end
    return checksums
end

"""
    save_fiber_properties(data_path, metadata_path, properties; overwrite=false)

Save a `FiberProperties` value to an explicitly selected JLD2 data path and
TOML metadata path. Existing paths are never replaced unless `overwrite=true`
is supplied explicitly.
"""
function save_fiber_properties(data_path::AbstractString,
                               metadata_path::AbstractString,
                               properties::FiberProperties;
                               overwrite::Bool=false)
    payload_checksum = fiber_properties_checksum(properties)
    metadata = Dict{String,Any}(
        "format" => _MMGNLSE_CACHE_FORMAT,
        "schema_version" => MMGNLSE_CACHE_SCHEMA_VERSION,
        "checksum_algorithm" => _MMGNLSE_CACHE_CHECKSUM_ALGORITHM,
        "canonical_value_format" => _CANONICAL_VALUE_FORMAT,
        "kind" => "FiberProperties",
        "payload_checksum" => payload_checksum,
        "arrays" => _fiber_array_checksums(properties),
        "provenance" => Dict{String,Any}(
            "beta_order" => beta_order(properties.beta),
            "num_modes" => length(properties.modes.beta0),
            "mode_field_shape" => collect(size(properties.modes.fields)),
            "wavelength" => Float64(properties.modes.wavelength),
        ),
    )
    writer = function (path)
        modes = properties.modes
        JLD2.jldopen(path, "w") do file
            file["beta_coefficients"] = properties.beta.coefficients
            file["raman"] = properties.raman
            file["mode_x"] = modes.x
            file["mode_y"] = modes.y
            file["mode_fields"] = modes.fields
            file["mode_beta0"] = modes.beta0
            file["mode_labels"] = modes.labels
            file["mode_wavelength"] = modes.wavelength
            file["config"] = properties.config
            file["n2"] = properties.n2
            file["omega0"] = properties.omega0
            if properties.S isa SpatialOverlap
                file["S_kind"] = "SpatialOverlap"
                file["S_values"] = properties.S.values
            else
                cp = properties.S
                file["S_kind"] = "MMGNLSECPDecomposition"
                file["S_weights"] = cp.λ
                for index in 1:4
                    file["S_factor_$index"] = cp.U[index]
                end
                file["S_layout"] = String(cp.metadata.layout)
                file["S_nmodes"] = cp.metadata.nmodes
                file["S_npolarizations"] = cp.metadata.npolarizations
                file["S_source_checksum"] = cp.metadata.source_checksum
                file["S_relative_error"] = cp.relative_error
                file["S_seed"] = cp.seed
                file["S_iterations"] = cp.iterations
                file["S_converged"] = cp.converged
                file["S_zero_tensor"] = cp.zero_tensor
            end
        end
    end
    return _write_cache_pair(data_path, metadata_path, metadata, writer; overwrite)
end

function _load_fiber_payload(data_path::AbstractString)
    data = JLD2.load(data_path)
    required = ("beta_coefficients", "raman", "mode_x", "mode_y",
                "mode_fields", "mode_beta0", "mode_labels", "mode_wavelength",
                "config", "n2", "omega0")
    all(key -> haskey(data, key), required) || throw(ArgumentError(
        "FiberProperties JLD2 payload is missing required datasets."))
    modes = FiberModeData(data["mode_x"], data["mode_y"], data["mode_fields"],
                          data["mode_beta0"], data["mode_labels"],
                          data["mode_wavelength"])
    overlap_kind = get(data, "S_kind",
                       haskey(data, "S_values") ? "SpatialOverlap" : "")
    overlap = if overlap_kind == "SpatialOverlap"
        haskey(data, "S_values") || throw(ArgumentError(
            "FiberProperties dense-overlap payload is missing S_values."))
        SpatialOverlap(data["S_values"])
    elseif overlap_kind == "MMGNLSECPDecomposition"
        cp_required = ("S_weights", "S_factor_1", "S_factor_2",
                       "S_factor_3", "S_factor_4", "S_layout", "S_nmodes",
                       "S_npolarizations", "S_source_checksum",
                       "S_relative_error", "S_seed", "S_iterations",
                       "S_converged", "S_zero_tensor")
        all(key -> haskey(data, key), cp_required) || throw(ArgumentError(
            "FiberProperties CP-overlap payload is incomplete."))
        MMGNLSECPDecomposition(
            data["S_weights"],
            ntuple(index -> data["S_factor_$index"], 4);
            layout=Symbol(data["S_layout"]),
            nmodes=Int(data["S_nmodes"]),
            npolarizations=Int(data["S_npolarizations"]),
            source_checksum=data["S_source_checksum"],
            relative_error=Float64(data["S_relative_error"]),
            seed=Int(data["S_seed"]),
            iterations=Int(data["S_iterations"]),
            converged=Bool(data["S_converged"]),
            zero_tensor=Bool(data["S_zero_tensor"]))
    else
        throw(ArgumentError(
            "Unsupported FiberProperties overlap kind $overlap_kind."))
    end
    return FiberProperties(TaylorBeta(data["beta_coefficients"]),
                           overlap, data["raman"],
                           modes, data["config"], data["n2"], data["omega0"])
end

"""
    load_fiber_properties(data_path, metadata_path;
                          source_S=nothing, expected_checksum=nothing)

Load and fully validate an explicit FiberProperties cache. If `source_S` is
provided, its canonical checksum must match the cached overlap tensor.
`expected_checksum` may be used to pin the complete logical payload.
"""
function load_fiber_properties(data_path::AbstractString,
                               metadata_path::AbstractString;
                               source_S=nothing,
                               expected_checksum=nothing)
    metadata = _read_cache_metadata(data_path, metadata_path, "FiberProperties")
    properties = _load_fiber_payload(data_path)
    arrays = get(metadata, "arrays", nothing)
    arrays isa AbstractDict || throw(ArgumentError(
        "FiberProperties metadata is missing array checksums."))
    for (name, checksum) in _fiber_array_checksums(properties)
        _require_checksum_match(checksum, get(arrays, name, nothing),
                                "FiberProperties $name")
    end
    payload_checksum = fiber_properties_checksum(properties)
    _require_checksum_match(payload_checksum, get(metadata, "payload_checksum", nothing),
                            "FiberProperties payload")
    if expected_checksum !== nothing
        _require_checksum_match(payload_checksum, expected_checksum,
                                "expected FiberProperties payload")
    end
    if source_S !== nothing
        source_S isa AbstractArray{<:Number,4} || throw(ArgumentError(
            "source_S must be a four-dimensional numeric overlap tensor."))
        source_checksum = canonical_array_checksum(source_S)
        _require_checksum_match(source_checksum, get(arrays, "S", nothing),
                                "FiberProperties source S")
    end
    return properties
end

function _cp_with_source_checksum(cp::MMGNLSECPDecomposition,
                                  source_checksum::AbstractString)
    return MMGNLSECPDecomposition(cp.λ, cp.U;
        layout=cp.metadata.layout,
        nmodes=cp.metadata.nmodes,
        npolarizations=cp.metadata.npolarizations,
        source_checksum=source_checksum,
        relative_error=cp.relative_error,
        seed=cp.seed,
        iterations=cp.iterations,
        converged=cp.converged,
        zero_tensor=cp.zero_tensor)
end

function _validated_cached_cp(cp::MMGNLSECPDecomposition, source_S)
    if source_S === nothing
        cp.metadata.source_checksum === nothing && throw(ArgumentError(
            "A CP cache requires source_S or a decomposition carrying a source checksum."))
        _require_sha256(cp.metadata.source_checksum, "CP source checksum")
        return cp
    end
    source_S isa AbstractArray{<:Number,4} || throw(ArgumentError(
        "source_S must be a four-dimensional numeric overlap tensor."))
    size(source_S) == cp.metadata.source_size || throw(DimensionMismatch(
        "source_S has size $(size(source_S)); CP metadata expects $(cp.metadata.source_size)."))
    source_checksum = canonical_array_checksum(source_S)
    stored = cp.metadata.source_checksum
    stored === nothing || stored == source_checksum || throw(ArgumentError(
        "The CP decomposition source checksum does not match source_S."))
    return stored === nothing ? _cp_with_source_checksum(cp, source_checksum) : cp
end

function _cp_logical_value(cp::MMGNLSECPDecomposition)
    return (
        weights=cp.λ,
        factors=cp.U,
        layout=cp.metadata.layout,
        source_size=cp.metadata.source_size,
        nmodes=cp.metadata.nmodes,
        npolarizations=cp.metadata.npolarizations,
        source_checksum=cp.metadata.source_checksum,
        relative_error=cp.relative_error,
        seed=cp.seed,
        iterations=cp.iterations,
        converged=cp.converged,
        zero_tensor=cp.zero_tensor,
    )
end

"""Return the deterministic logical checksum stored for a CP decomposition."""
cp_decomposition_checksum(cp::MMGNLSECPDecomposition) =
    _canonical_value_checksum(_cp_logical_value(cp))

function _cp_array_checksums(cp::MMGNLSECPDecomposition)
    checksums = Dict{String,Any}(
        "weights" => canonical_array_checksum(cp.λ),
    )
    for index in 1:4
        checksums["factor_$index"] = canonical_array_checksum(cp.U[index])
    end
    return checksums
end

"""
    save_cp_decomposition(data_path, metadata_path, cp;
                          source_S=nothing, overwrite=false)

Save a CP decomposition and deterministic provenance to explicit JLD2/TOML
paths. Passing `source_S` verifies (and, when necessary, attaches) the source
tensor checksum before any files are written.
"""
function save_cp_decomposition(data_path::AbstractString,
                               metadata_path::AbstractString,
                               cp::MMGNLSECPDecomposition;
                               source_S=nothing,
                               overwrite::Bool=false)
    cached_cp = _validated_cached_cp(cp, source_S)
    metadata = Dict{String,Any}(
        "format" => _MMGNLSE_CACHE_FORMAT,
        "schema_version" => MMGNLSE_CACHE_SCHEMA_VERSION,
        "checksum_algorithm" => _MMGNLSE_CACHE_CHECKSUM_ALGORITHM,
        "canonical_value_format" => _CANONICAL_VALUE_FORMAT,
        "kind" => "MMGNLSECPDecomposition",
        "payload_checksum" => cp_decomposition_checksum(cached_cp),
        "source_checksum" => cached_cp.metadata.source_checksum,
        "arrays" => _cp_array_checksums(cached_cp),
        "provenance" => Dict{String,Any}(
            "layout" => String(cached_cp.metadata.layout),
            "source_size" => collect(cached_cp.metadata.source_size),
            "nmodes" => cached_cp.metadata.nmodes,
            "npolarizations" => cached_cp.metadata.npolarizations,
            "rank" => cp_rank(cached_cp),
            "relative_error" => cached_cp.relative_error,
            "seed" => cached_cp.seed,
            "iterations" => cached_cp.iterations,
            "converged" => cached_cp.converged,
            "zero_tensor" => cached_cp.zero_tensor,
        ),
    )
    writer = path -> JLD2.jldsave(path;
        weights=cached_cp.λ,
        factor_1=cached_cp.U[1],
        factor_2=cached_cp.U[2],
        factor_3=cached_cp.U[3],
        factor_4=cached_cp.U[4])
    return _write_cache_pair(data_path, metadata_path, metadata, writer; overwrite)
end

function _load_cp_payload(data_path::AbstractString, metadata)
    data = JLD2.load(data_path)
    required = ("weights", "factor_1", "factor_2", "factor_3", "factor_4")
    all(key -> haskey(data, key), required) || throw(ArgumentError(
        "CP JLD2 payload is missing required datasets."))
    provenance = get(metadata, "provenance", nothing)
    provenance isa AbstractDict || throw(ArgumentError(
        "CP metadata is missing provenance."))
    source_size = Tuple(Int.(get(provenance, "source_size", Int[])))
    length(source_size) == 4 || throw(ArgumentError(
        "CP provenance source_size must contain four dimensions."))
    factors = ntuple(index -> data["factor_$index"], 4)
    all(index -> size(factors[index], 1) == source_size[index], 1:4) ||
        throw(DimensionMismatch("CP factors do not match provenance source_size."))
    length(data["weights"]) == Int(get(provenance, "rank", -1)) ||
        throw(DimensionMismatch("CP weights do not match provenance rank."))
    source_checksum = _require_sha256(get(metadata, "source_checksum", nothing),
                                      "CP source checksum")
    return MMGNLSECPDecomposition(data["weights"], factors;
        layout=Symbol(get(provenance, "layout", "")),
        nmodes=Int(get(provenance, "nmodes", 0)),
        npolarizations=Int(get(provenance, "npolarizations", 0)),
        source_checksum=source_checksum,
        relative_error=Float64(get(provenance, "relative_error", NaN)),
        seed=Int(get(provenance, "seed", 0)),
        iterations=Int(get(provenance, "iterations", 0)),
        converged=Bool(get(provenance, "converged", false)),
        zero_tensor=Bool(get(provenance, "zero_tensor", false)))
end

"""
    load_cp_decomposition(data_path, metadata_path;
                          source_S=nothing, expected_checksum=nothing)

Load a CP cache after verifying the JLD2 file, every factor checksum, logical
payload provenance, and (when supplied) the exact dense source tensor.
"""
function load_cp_decomposition(data_path::AbstractString,
                               metadata_path::AbstractString;
                               source_S=nothing,
                               expected_checksum=nothing)
    metadata = _read_cache_metadata(
        data_path, metadata_path, "MMGNLSECPDecomposition")
    cp = _load_cp_payload(data_path, metadata)
    arrays = get(metadata, "arrays", nothing)
    arrays isa AbstractDict || throw(ArgumentError(
        "CP metadata is missing array checksums."))
    for (name, checksum) in _cp_array_checksums(cp)
        _require_checksum_match(checksum, get(arrays, name, nothing), "CP $name")
    end
    payload_checksum = cp_decomposition_checksum(cp)
    _require_checksum_match(payload_checksum, get(metadata, "payload_checksum", nothing),
                            "CP payload")
    if expected_checksum !== nothing
        _require_checksum_match(payload_checksum, expected_checksum,
                                "expected CP payload")
    end
    if source_S !== nothing
        source_S isa AbstractArray{<:Number,4} || throw(ArgumentError(
            "source_S must be a four-dimensional numeric overlap tensor."))
        validate_cp_source(cp, source_S; checksum_fn=canonical_array_checksum)
    end
    return cp
end
