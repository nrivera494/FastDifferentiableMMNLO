struct AdjointGradient{W,N,K}
    wrt::W
    normalization::N
    kwargs::K
end
AdjointGradient(; wrt=InitialField(), normalization=PhotonNormalized(), kwargs...) =
    AdjointGradient(wrt, normalization, NamedTuple(kwargs))

struct AutomaticDifferentiationGradient{W,N,K}
    wrt::W
    normalization::N
    kwargs::K
end
AutomaticDifferentiationGradient(; wrt=InitialField(), normalization=PowerNormalized(), kwargs...) =
    AutomaticDifferentiationGradient(wrt, normalization, NamedTuple(kwargs))

Base.@kwdef struct FiniteDifferenceGradient
    method::Symbol = :central
    relstep::Float64 = 1e-5
    absstep::Float64 = 1e-8
end

struct NoGradient end

Base.@kwdef struct DifferentiableRegularizer{V,G}
    value::V
    gradient::G
end

Base.@kwdef struct PropagationOptimizationProblem{P,B,O,G,PB,PF,R}
    parameters::P
    build_problem::B
    objective::O
    sense::Symbol = :minimize
    gradient::G = nothing
    pullback::PB = nothing
    pushforward::PF = nothing
    regularizers::R = ()
end

Base.@kwdef struct GradientDescent
    stepsize::Float64 = 1e-2
    maxiters::Int = 100
    tol::Float64 = 1e-8
end

Base.@kwdef struct CoordinateSearch
    initial_step::Float64 = 1e-2
    shrink::Float64 = 0.5
    maxiters::Int = 100
    tol::Float64 = 1e-8
end

Base.@kwdef struct OptimizationResult{P,T,H}
    parameters::P
    objective_value::T
    loss::T
    iterations::Int
    converged::Bool
    history::H
end

function optimization_value(opt::PropagationOptimizationProblem,
                            θ=opt.parameters)
    problem = opt.build_problem(θ)
    sol = solve(problem)
    return objective_value(opt.objective, problem, sol)
end

value(opt::PropagationOptimizationProblem, θ=opt.parameters) =
    optimization_value(opt, θ)

function objective_value(objective::AbstractObservable,
                         problem::PulsePropagationProblem,
                         sol::PulseTrajectory)
    return value(objective, sol)
end

function objective_value(objective::VarianceObjective,
                         problem::PulsePropagationProblem,
                         sol::PulseTrajectory)
    return value(objective, problem, sol)
end

function objective_value(objective::Function,
                         problem::PulsePropagationProblem,
                         sol::PulseTrajectory)
    return objective(sol)
end

function optimization_loss(opt::PropagationOptimizationProblem,
                           θ=opt.parameters)
    return _objective_evaluation(opt, θ).loss
end

function _objective_evaluation(opt::PropagationOptimizationProblem, θ)
    problem = opt.build_problem(θ)
    sol = solve(problem)
    J = objective_value(opt.objective, problem, sol)
    L = _loss_from_objective(opt, θ, J)
    return (; problem, sol, objective_value=J, loss=L)
end

_loss_from_objective(opt::PropagationOptimizationProblem, θ, J) =
    _sense_sign(opt.sense) * J + regularization_value(opt, θ)

function regularization_value(opt::PropagationOptimizationProblem, θ)
    isempty(opt.regularizers) && return zero(real(eltype(θ)))
    return sum(_regularizer_value(r, θ) for r in opt.regularizers)
end

_regularizer_value(r::DifferentiableRegularizer, θ) = r.value(θ)
_regularizer_value(r::Function, θ) = r(θ)

function optimization_gradient(opt::PropagationOptimizationProblem,
                               θ=opt.parameters)
    return _gradient_evaluation(opt, θ).gradient
end

gradient(opt::PropagationOptimizationProblem, θ=opt.parameters) =
    optimization_gradient(opt, θ)

function value_and_gradient(opt::PropagationOptimizationProblem,
                            θ=opt.parameters)
    eval = _gradient_evaluation(opt, θ)
    return eval.objective_value, eval.gradient
end

function _adjoint_parameter_gradient(opt, θ, backend::AdjointGradient)
    return _adjoint_parameter_gradient_evaluation(opt, θ, backend).gradient
end

function _gradient_evaluation(opt::PropagationOptimizationProblem, θ)
    backend = opt.gradient
    backend === nothing &&
        error("No gradient backend is configured. Use FiniteDifferenceGradient(), AdjointGradient(), or a gradient-free optimizer.")
    if backend isa FiniteDifferenceGradient
        eval = _objective_evaluation(opt, θ)
        g = _finite_difference_gradient(θ -> optimization_loss(opt, θ), θ, backend;
                                        fθ=eval.loss)
        return merge(eval, (; gradient=g))
    elseif backend isa AdjointGradient
        return _adjoint_parameter_gradient_evaluation(opt, θ, backend)
    elseif backend isa AutomaticDifferentiationGradient
        return _ad_parameter_gradient_evaluation(opt, θ, backend)
    else
        error("Unsupported gradient backend $(typeof(backend)).")
    end
end

function _adjoint_parameter_gradient_evaluation(opt, θ, backend::AdjointGradient)
    opt.objective isa VarianceObjective &&
        return _variance_parameter_gradient_evaluation(opt, θ, backend)
    opt.pullback === nothing &&
        error("AdjointGradient requires a pullback from initial-field gradients to parameter gradients.")
    problem = opt.build_problem(θ)
    sol = solve(problem)
    grad = gradient(problem, opt.objective;
                    trajectory=sol,
                    method=Adjoint(),
                    wrt=backend.wrt,
                    normalization=backend.normalization,
                    backend.kwargs...)
    gθ = opt.pullback(θ, problem, sol, grad.initial_field_gradient, grad.value)
    gradient_value = _sense_sign(opt.sense) .* gθ .+ regularization_gradient(opt, θ)
    L = _loss_from_objective(opt, θ, grad.value)
    return (; problem, sol, objective_value=grad.value, loss=L,
            gradient=gradient_value)
end

function _variance_parameter_gradient(opt, θ, backend::AdjointGradient)
    return _variance_parameter_gradient_evaluation(opt, θ, backend).gradient
end

function _variance_parameter_gradient_evaluation(opt, θ, backend::AdjointGradient)
    opt.pushforward === nothing &&
        error("AdjointGradient for VarianceObjective requires a pushforward from parameters to initial Fourier-field directions. Use FiniteDifferenceGradient() or set pushforward.")
    problem = opt.build_problem(θ)
    sol = solve(problem)
    J = value(opt.objective, problem, sol)
    directions = applicable(opt.pushforward, θ, problem, sol) ?
                 opt.pushforward(θ, problem, sol) : nothing
    gθ = zeros(real(eltype(θ)), size(θ))
    for i in eachindex(θ)
        direction = directions === nothing ?
                    opt.pushforward(θ, problem, sol, i) :
                    _parameter_direction(directions, i)
        d = variance_directional_derivative(
            opt.objective, problem, direction;
            trajectory=sol,
            backend.kwargs...)
        gθ[i] = d.directional_derivative
    end
    gradient_value = _sense_sign(opt.sense) .* gθ .+ regularization_gradient(opt, θ)
    return (; problem, sol, objective_value=J, loss=_loss_from_objective(opt, θ, J),
            gradient=gradient_value)
end

function _parameter_direction(directions::AbstractVector, i)
    return directions[i]
end

function _parameter_direction(directions::AbstractArray, i)
    ndims(directions) >= 3 ||
        error("pushforward array directions must have field dimensions plus a parameter dimension.")
    return selectdim(directions, ndims(directions), i)
end

function _ad_parameter_gradient(opt, θ, backend::AutomaticDifferentiationGradient)
    return _ad_parameter_gradient_evaluation(opt, θ, backend).gradient
end

function _ad_parameter_gradient_evaluation(opt, θ, backend::AutomaticDifferentiationGradient)
    opt.objective isa VarianceObjective &&
        error("AutomaticDifferentiationGradient for VarianceObjective requires differentiating the variance-gradient calculation. Use FiniteDifferenceGradient() for now.")
    opt.pullback === nothing &&
        error("AutomaticDifferentiationGradient requires a pullback from initial-field gradients to parameter gradients.")
    problem = opt.build_problem(θ)
    sol = solve(problem)
    grad = gradient(problem, opt.objective;
                    trajectory=sol,
                    method=AutomaticDifferentiation(),
                    wrt=backend.wrt,
                    normalization=backend.normalization,
                    backend.kwargs...)
    gθ = opt.pullback(θ, problem, sol, grad.initial_field_gradient, grad.value)
    gradient_value = _sense_sign(opt.sense) .* gθ .+ regularization_gradient(opt, θ)
    L = _loss_from_objective(opt, θ, grad.value)
    return (; problem, sol, objective_value=grad.value, loss=L,
            gradient=gradient_value)
end

function regularization_gradient(opt::PropagationOptimizationProblem, θ)
    isempty(opt.regularizers) && return zero.(θ)
    g = zero.(θ)
    fd = FiniteDifferenceGradient()
    for r in opt.regularizers
        g .+= _regularizer_gradient(r, θ, fd)
    end
    return g
end

_regularizer_gradient(r::DifferentiableRegularizer, θ, fd) = r.gradient(θ)
_regularizer_gradient(r::Function, θ, fd) =
    _finite_difference_gradient(x -> r(x), θ, fd)

function _finite_difference_gradient(f, θ, spec::FiniteDifferenceGradient; fθ=nothing)
    g = zeros(real(eltype(θ)), size(θ))
    for i in eachindex(θ)
        h = spec.absstep + spec.relstep * max(abs(real(θ[i])), one(real(eltype(θ))))
        θp = copy(θ)
        θm = copy(θ)
        if spec.method === :forward
            θp[i] += h
            base = fθ === nothing ? f(θ) : fθ
            g[i] = (f(θp) - base) / h
        elseif spec.method === :central
            θp[i] += h
            θm[i] -= h
            g[i] = (f(θp) - f(θm)) / (2h)
        else
            error("FiniteDifferenceGradient.method must be :central or :forward.")
        end
    end
    return g
end

function optimize(opt::PropagationOptimizationProblem, method::GradientDescent;
                  maxiters=method.maxiters, stepsize=method.stepsize,
                  tol=method.tol)
    opt.gradient === nothing &&
        error("GradientDescent requires opt.gradient. Use FiniteDifferenceGradient(), AdjointGradient(), or CoordinateSearch().")
    θ = copy(opt.parameters)
    history = NamedTuple[]
    converged = false
    iter_done = 0
    last_eval = nothing
    for iter in 1:maxiters
        eval = _gradient_evaluation(opt, θ)
        g = eval.gradient
        gnorm = norm(g)
        push!(history, (; iteration=iter, objective=eval.objective_value,
                        loss=eval.loss, gradnorm=gnorm, stepsize=stepsize))
        iter_done = iter
        last_eval = eval
        if gnorm < tol
            converged = true
            break
        end
        θ .= θ .- stepsize .* g
        last_eval = nothing
    end
    final_eval = last_eval === nothing ? _objective_evaluation(opt, θ) : last_eval
    return OptimizationResult(; parameters=θ,
                              objective_value=final_eval.objective_value,
                              loss=final_eval.loss,
                              iterations=iter_done,
                              converged=converged,
                              history=history)
end

function optimize(opt::PropagationOptimizationProblem, method::CoordinateSearch;
                  maxiters=method.maxiters, initial_step=method.initial_step,
                  shrink=method.shrink, tol=method.tol)
    θ = copy(opt.parameters)
    step = initial_step
    best_eval = _objective_evaluation(opt, θ)
    best = best_eval.loss
    history = NamedTuple[]
    converged = false
    iter_done = 0
    for iter in 1:maxiters
        improved = false
        for i in eachindex(θ)
            for sgn in (-1, 1)
                trial = copy(θ)
                trial[i] += sgn * step
                trial_eval = _objective_evaluation(opt, trial)
                L = trial_eval.loss
                if L < best
                    θ = trial
                    best = L
                    best_eval = trial_eval
                    improved = true
                end
            end
        end
        push!(history, (; iteration=iter, objective=best_eval.objective_value,
                        loss=best, step=step))
        iter_done = iter
        if !improved
            step *= shrink
        end
        if step < tol
            converged = true
            break
        end
    end
    return OptimizationResult(; parameters=θ,
                              objective_value=best_eval.objective_value,
                              loss=best_eval.loss,
                              iterations=iter_done,
                              converged=converged,
                              history=history)
end

function optimize(opt::PropagationOptimizationProblem, method::Optim.AbstractOptimizer;
                  options=Optim.Options(), kwargs...)
    x0, template = _optim_parameter_vector(opt.parameters)
    evaluator = _optim_cached_evaluator(opt, template)
    if method isa Optim.FirstOrderOptimizer
        opt.gradient === nothing &&
            error("$(typeof(method)) requires opt.gradient. Use AdjointGradient(), AutomaticDifferentiationGradient(), or FiniteDifferenceGradient().")
        f = x -> evaluator.loss(x)
        g = x -> evaluator.gradient(x)
        fg = x -> begin
            eval = evaluator.loss_and_gradient(x)
            return eval.loss, eval.gradient
        end
        od = Optim.OnceDifferentiable(f, g, fg, x0, NaN, similar(x0);
                                      inplace=false)
        optim_result = Optim.optimize(od, x0, method, options; kwargs...)
    else
        f = x -> evaluator.loss(x)
        optim_result = Optim.optimize(f, x0, method, options; kwargs...)
    end

    θ = _optim_unflatten(Optim.minimizer(optim_result), template)
    final_eval = evaluator.objective(Optim.minimizer(optim_result))
    return OptimizationResult(; parameters=θ,
                              objective_value=final_eval.objective_value,
                              loss=final_eval.loss,
                              iterations=Optim.iterations(optim_result),
                              converged=Optim.converged(optim_result),
                              history=(; optim_result))
end

function _optim_parameter_vector(θ)
    θ isa AbstractArray ||
        error("Optim.jl backends require opt.parameters to be a real AbstractArray.")
    eltype(θ) <: Real ||
        error("Optim.jl backends require real-valued parameters; got eltype $(eltype(θ)).")
    return Float64.(vec(copy(θ))), copy(θ)
end

function _optim_unflatten(x, template)
    return reshape(copy(x), size(template))
end

function _optim_cached_evaluator(opt, template)
    last_x = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    last_eval = Ref{Any}(nothing)
    last_has_gradient = Ref(false)

    same_x = x -> begin
        cached = last_x[]
        cached !== nothing && length(cached) == length(x) && all(cached .== x)
    end
    θ_from_x = x -> _optim_unflatten(x, template)

    eval_at = function (x; gradient::Bool=false)
        if same_x(x) && (!gradient || last_has_gradient[])
            return last_eval[]
        end
        θ = θ_from_x(x)
        eval = gradient ? _gradient_evaluation(opt, θ) : _objective_evaluation(opt, θ)
        last_x[] = Float64.(vec(copy(x)))
        last_eval[] = eval
        last_has_gradient[] = gradient
        return eval
    end

    return (; objective=x -> eval_at(x),
            loss=x -> eval_at(x).loss,
            gradient=x -> eval_at(x; gradient=true).gradient,
            loss_and_gradient=x -> eval_at(x; gradient=true))
end

function _sense_sign(sense::Symbol)
    sense === :minimize && return 1.0
    sense === :maximize && return -1.0
    error("sense must be :minimize or :maximize.")
end
