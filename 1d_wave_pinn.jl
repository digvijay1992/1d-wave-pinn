using Pkg
Pkg.activate(".")

if !isfile(joinpath(@__DIR__, "Project.toml"))
    Pkg.activate(@__DIR__)
    Pkg.add([
        "NeuralPDE",
        "Lux",
        "ModelingToolkit",
        "Optimization",
        "OptimizationOptimisers",
        "OptimizationOptimJL",
        "Random",
        "ComponentArrays",
        "Plots",
        "JLD2",
        "CSV",
        "Tables",
        "DomainSets",
        "Statistics"
    ])
end

using NeuralPDE
using Lux
using ModelingToolkit
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Random
using ComponentArrays
using Plots
using JLD2
using CSV, Tables
using DomainSets
using Statistics

# ============================================================
# 1D Wave Equation with Physics-Informed Neural Networks
#
# PDE:
#   u_tt = c^2 u_xx,   x in [0,1], t in [0,T]
#
# Boundary conditions (BCs):
#   u(0,t) = 0
#   u(1,t) = 0
#
# Initial conditions (ICs):
#   u(x,0)   = x(1-x)
#   u_t(x,0) = 0
#
# Grid used for evaluation:
#   dx = 0.1
# ============================================================

# -----------------------------
# Parameters
# -----------------------------
const c        = 1.0
const Tfinal   = 1.0
const dx       = 0.1
const dt       = 0.02
const Nfourier = 200
const seed     = 1234

Random.seed!(seed)

# -----------------------------
# Analytical solution
# -----------------------------
# For f(x) = x(1-x), the sine series on [0,1] has coefficients:
#   a_n = 2 ∫_0^1 x(1-x) sin(nπx) dx
# With zero initial velocity:
#   u(x,t) = Σ a_n cos(c n π t) sin(n π x)
#
# For this particular f(x), only odd n are nonzero:
#   a_n = 8 / (π^3 n^3), n odd
#   a_n = 0           , n even

function fourier_coeff(n::Int)
    isodd(n) ? 8.0 / (pi^3 * n^3) : 0.0
end

function u_true(x, t; c = 1.0, N = 200)
    s = 0.0
    for n in 1:N
        an = fourier_coeff(n)
        s += an * cos(c * n * pi * t) * sin(n * pi * x)
    end
    return s
end

u_true_grid(xs, ts; c = 1.0, N = 200) = [u_true(x, t; c=c, N=N) for x in xs, t in ts]

# -----------------------------
# PDE definition (symbolic)
# -----------------------------
@parameters x t
@variables u(..)

Dx  = Differential(x)
Dt  = Differential(t)
Dxx = Dx^2
Dtt = Dt^2

eq = Dtt(u(x, t)) ~ c^2 * Dxx(u(x, t))

bcs = [
    u(0, t) ~ 0.0,
    u(1, t) ~ 0.0,
    u(x, 0) ~ x * (1 - x),
    Dt(u(x, 0)) ~ 0.0
]

domains = [
    x ∈ Interval(0.0, 1.0),
    t ∈ Interval(0.0, Tfinal)
]

@named pdesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

# -----------------------------
# Neural network model
# -----------------------------
# One network takes (x, t) as input and predicts u(x,t).

rng = Random.default_rng()

chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 1)
)

ps, st = Lux.setup(rng, chain)
ps = ComponentArray(ps)
ps = ComponentArray(map(Float64, ps))  # promote to Float64

# -----------------------------
# PINN discretization and training
# -----------------------------
# We use a grid-based training strategy with spacing dx.
strategy = GridTraining(dx)

discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
prob = NeuralPDE.discretize(pdesys, discretization)

# Record loss history for plotting later
loss_history = Float64[]
phase_history = String[]

# Callback to record loss with iteration and show progress
function callback(p, l; phase)
    push!(loss_history, l)
    push!(phase_history, phase)
    println("[$phase] current loss = $l")
    return false
end

# 1) Adam phase
res_adam = Optimization.solve(
    prob,
    OptimizationOptimisers.Adam(0.001);
    maxiters = 4000,
    callback = (p, l) -> callback(p, l; phase = "Adam")
)

# 2) BFGS phase (refinement)
res_bfgs = Optimization.solve(
    prob,
    OptimizationOptimJL.BFGS();
    maxiters = 3000,
    u0 = res_adam.u,
    callback = (p, l) -> callback(p, l; phase = "BFGS")
)

θ = res_bfgs.u

# -----------------------------
# PINN predictor (numerical solution)
# -----------------------------
phi = discretization.phi

function u_pinn(xval, tval, θ)
    X = reshape([xval, tval], 2, 1)
    y = first(phi(X, θ))
    return y[1]
end

function u_pinn_grid(xs, ts, θ)
    [u_pinn(x, t, θ) for x in xs, t in ts]
end

# -----------------------------
# Evaluation grid
# -----------------------------
x_vals = collect(0.0:dx:1.0)
t_vals = collect(0.0:dt:Tfinal)

U_true = u_true_grid(x_vals, t_vals; c = c, N = Nfourier)
U_pinn = u_pinn_grid(x_vals, t_vals, θ)
Err    = abs.(U_pinn .- U_true)

# -----------------------------
# Error metrics
# -----------------------------
mae   = mean(Err)
rmse  = sqrt(mean((U_pinn .- U_true).^2))
maxerr = maximum(Err)

println("==========================================")
println("1D Wave Equation PINN Results")
println("==========================================")
println("MAE   = ", mae)
println("RMSE  = ", rmse)
println("MaxErr= ", maxerr)

selected_times = [0.0, 0.25, 0.5, 0.75, 1.0]
println("\nErrors at selected time snapshots:")
for tsnap in selected_times
    j = argmin(abs.(t_vals .- tsnap))
    e_snapshot = abs.(U_pinn[:, j] .- U_true[:, j])
    println("t = $(round(t_vals[j], digits=3)) : ",
            "MAE = $(mean(e_snapshot)), ",
            "MaxErr = $(maximum(e_snapshot))")
end

# ============================================================
# Visualization (static and GIF)
# ============================================================
default(size=(900, 600), lw=2)

# 1) Static snapshots for report
j0 = argmin(abs.(t_vals .- 0.0))
p1 = plot(
    x_vals, U_true[:, j0],
    label = "True",
    xlabel = "x",
    ylabel = "u(x,t)",
    title = "Initial Condition (t = 0)",
    legend = :topright
)
plot!(p1, x_vals, U_pinn[:, j0], label = "PINN", ls = :dash)


jend = length(t_vals)
p2 = plot(
    x_vals, U_true[:, jend],
    label = "True",
    xlabel = "x",
    ylabel = "u(x,t)",
    title = "Solution at t = $(round(t_vals[jend], digits=2))",
    legend = :topright
)
plot!(p2, x_vals, U_pinn[:, jend], label = "PINN", ls = :dash)


p3 = heatmap(
    t_vals, x_vals, U_true,
    xlabel = "t",
    ylabel = "x",
    title = "True Solution",
    colorbar_title = "u"
)

p4 = heatmap(
    t_vals, x_vals, U_pinn,
    xlabel = "t",
    ylabel = "x",
    title = "PINN Solution",
    colorbar_title = "u"
)

plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1200))
savefig("wave_pinn_comparison.png")

savefig(p1, "initial_condition_comparison.png")
savefig(p2, "finaltime_comparison.png")
savefig(p3, "true_solution_heatmap.png")
savefig(p4, "pinn_solution_heatmap.png")

# -----------------------------
# GIF 1: line plot (PINN vs True) over time
# -----------------------------
# Use at most ~20 frames by sampling every k-th time step
# (adjust step to trade off speed vs smoothness)
step_line = max(1, Int(cld(length(t_vals), 20)))  # aim for ≤ 20 frames

anim_pinn_vs_true = @animate for k in 1:step_line:length(t_vals)
    t = t_vals[k]
    plot(
        x_vals, U_true[:, k],
        label = "True",
        xlabel = "x",
        ylabel = "u(x,t)",
        title = "True vs PINN (t = $(round(t, digits=2)))",
        ylim = (minimum(U_true), maximum(U_true)),
        legend = :topright
    )
    plot!(x_vals, U_pinn[:, k], label = "PINN", ls = :dash)
end

gif(anim_pinn_vs_true, "pinn_vs_true.gif"; fps = 5)

# -----------------------------
# GIF 2: loss history vs iteration
#       (Adam and BFGS clearly separated)
#       → Downsample iterations to save time
# -----------------------------
using StatsBase

# Compute indices where phase changes from Adam to BFGS
adam_iters = findall(==("Adam"), phase_history)
bfgs_iters = findall(==("BFGS"), phase_history)

# Downsample loss history to at most ~100 points
n_loss = length(loss_history)
step_loss = max(1, Int(cld(n_loss, 100)))
loss_idx = collect(1:step_loss:n_loss)

anim_loss = @animate for i in loss_idx
    plot(
        1:i, loss_history[1:i],
        yscale = :log10,
        xlabel = "Iteration",
        ylabel = "Loss",
        title = "Loss History (Adam + BFGS)",
        label = "Loss"
    )
    # vertical line at end of Adam phase
    if !isempty(adam_iters)
        adam_end = maximum(adam_iters)
        vline!([adam_end]; label = "Adam → BFGS", ls = :dash, lc = :red)
    end
end

gif(anim_loss, "loss_history.gif"; fps = 5)

# -----------------------------
# GIF 3: heat-map (True vs PINN)
#       Layout: 2 rows, 1 column
#       → Use at most ~20 frames
# -----------------------------
min_u = minimum([minimum(U_true), minimum(U_pinn)])
max_u = maximum([maximum(U_true), maximum(U_pinn)])

step_heat = max(1, Int(cld(length(t_vals), 20)))  # again, ≤ 20 frames

anim_heat = @animate for i in 1:step_heat:length(t_vals)
    ti = t_vals[i]
    p_true = heatmap(
        t_vals[1:i], x_vals, U_true[:, 1:i],
        xlabel = "t",
        ylabel = "x",
        clim = (min_u, max_u),
        title = "True solution (up to t = $(round(ti, digits=2)))",
        colorbar = false
    )
    p_pinn = heatmap(
        t_vals[1:i], x_vals, U_pinn[:, 1:i],
        xlabel = "t",
        ylabel = "x",
        clim = (min_u, max_u),
        title = "PINN solution (up to t = $(round(ti, digits=2)))",
        colorbar = true
    )
    plot(p_true, p_pinn, layout = (2, 1), size = (600, 600))
end

gif(anim_heat, "heatmap_pinn_vs_true.gif"; fps = 5)

# -----------------------------
# Save simulation data (JLD2)
# -----------------------------
@save "wave_pinn_simulation.jld2" x_vals t_vals U_true U_pinn Err c Tfinal dx dt Nfourier

# -----------------------------
# Optional: save as CSV for external tools
# -----------------------------
function build_table(xs, ts, U_true, U_pinn, Err)
    data = NamedTuple[]
    for (i, x) in enumerate(xs)
        for (j, t) in enumerate(ts)
            push!(data, (x = x,
                         t = t,
                         u_true = U_true[i, j],
                         u_pinn = U_pinn[i, j],
                         error = Err[i, j]))
        end
    end
    return data
end

data_table = build_table(x_vals, t_vals, U_true, U_pinn, Err)
CSV.write("wave_pinn_simulation.csv", Tables.columntable(data_table))