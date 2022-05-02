using Bioceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!

using NCDatasets
ds = Dataset("../../data/interim/input_coarse.nc")
const Nx, Ny, Nz = size(ds["b"])


grid = RectilinearGrid(CPU(),
    size=(Nx, Ny, Nz),
    x=(minimum(ds["xC"]), maximum(ds["xC"])),
    y=(minimum(ds["yC"]), maximum(ds["yC"])),
    z=ds["zF"][:],
    topology=(Periodic, Bounded, Bounded)
)

coriolis = FPlane(latitude=60)

const cᴰ = 1e-4 # quadratic drag coefficient

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

bottom_drag_bc_u = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
bottom_drag_bc_v = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_v)


@inline νh(x,y,z,t) = ifelse((y>-((Ny*10-20)/2)kilometers)&(y<((Ny*10-20)/2)kilometers), 10, 200)
horizontal_closure = HorizontalScalarDiffusivity(ν=νh, κ=νh)

@inline νz(x,y,z,t) = ifelse((y>-((Ny*10-20)/2)kilometers)&(y<((Ny*10-20)/2)kilometers), 1e-5, 1e-4)
vertical_closure = ScalarDiffusivity(ν=νz, κ=νz)


# ---- 
# ---- initial mixed-layer eddy velocities

# no penetration bc for mixed-layer eddy vertical velocity
no_penetration = ImpenetrableBoundaryCondition()
w_mle_bc = FieldBoundaryConditions(grid, (Center, Center, Face), top=no_penetration, bottom=no_penetration)
v_mle_bc = FieldBoundaryConditions(grid, (Center, Face, Center), north=no_penetration, south=no_penetration)

# initial zero fields for mixed-layer eddy velocities
u_mle = XFaceField(grid)
v_mle = YFaceField(grid, boundary_conditions=v_mle_bc)
w_mle = ZFaceField(grid, boundary_conditions=w_mle_bc)

# build AdvectiveForcing from mixed-layer eddy velocities
mle_forcing = AdvectiveForcing(WENO5(; grid), u = u_mle, v = v_mle, w = w_mle)


model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b),
                            buoyancy = BuoyancyTracer(),
                            forcing = (; b = mle_forcing),
                            boundary_conditions = (u=u_bcs, v=v_bcs))



bᵢ = Float64.(ds["b"][:])

set!(model; b=bᵢ)


b = model.tracers.b
f = model.coriolis.f

# shear operations
uz_op = @at((Face, Center, Center), -∂y(b) / f );
vz_op = @at((Center, Face, Center),  ∂x(b) / f );
# compute shear
uz = compute!(Field(uz_op))
vz = compute!(Field(vz_op))

# include function for cumulative integration
include("cumulative_vertical_integration.jl")

# compute geostrophic velocities
uᵢ = cumulative_vertical_integration!(uz)
vᵢ = cumulative_vertical_integration!(vz)

# prescribe geostrophic velocities for initial condition
set!(model; u = uᵢ, v = vᵢ)


simulation = Simulation(model, Δt = 1minutes, stop_time = 80day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1hour)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


h = Field{Center, Center, Nothing}(grid) 
# buoyancy decrease criterium for determining the mixed-layer depth
const g = 9.82 # gravity
const ρₒ = 1026 # reference density
const Δb = (g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = MixedLayerDepth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)



# ---- 
# ---- compute gradient of buoyancy

b = model.tracers.b
∂b∂x = Field(∂x(b))
∂b∂y = Field(∂y(b))
∂b∂z = Field(∂z(b))

function compute_∇b!(sim)
    compute!(∂b∂x)
    compute!(∂b∂y)
    compute!(∂b∂z)
    return nothing
end
simulation.callbacks[:compute_∇b] = Callback(compute_∇b!)

# ----
# ---- compute Ψₑ

# repeating the imports just to make it easier while moduling it
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ


@kernel function _compute_Ψₑ!(Ψₑ, grid, h, ∂ₕb, ∂b∂z, μ, f, ce, Lfₘ, ΔS, τ, minpoints, Vm)
    i, j = @index(Global, NTuple)

    
    # average ∇ₕb and N over the mixed layer
    
    ∂ₕb_sum = 0
    N_sum = 0
    Δz_sum = 0

    h_ij = @inbounds h[i, j]
    
    # number of z points in the mixed layer
    npoints = 0
    
    @unroll for k in grid.Nz : -1 : 1 # scroll from surface to bottom       
        z_center = znode(Center(), Face(), Face(), i, j, k, grid)

        if z_center > -h_ij
            npoints += 1
            
            Δz_ijk = Δzᶜᶜᶜ(i, j, k, grid)

            ∂ₕb_sum = ∂ₕb_sum + @inbounds ∂ₕb[i, j, k] * Δz_ijk
            N_sum = N_sum + @inbounds sqrt(max(zero(eltype(grid)), ∂b∂z[i, j, k])) * Δz_ijk 
            Δz_sum = Δz_sum + Δz_ijk

        end
    end

    ∂ₕbₘₗ = ∂ₕb_sum/Δz_sum
    Nₘₗ = N_sum/Δz_sum
    
    Lf = max(Nₘₗ*h_ij/abs(f), Lfₘ)
    
    # compute eddy stream function
    @unroll for k in grid.Nz : -1 : 1 # scroll to point just above the bottom       
        z_face = znode(Center(), Center(), Face(), i, j, k, grid)
        
        Ψ_max = @inbounds Δzᶜᶜᶠ(i, j, k, grid) * Vm
        Ψ_ijk = ce * (ΔS/Lf) * ((h_ij^2)/sqrt(f^2 + τ^-2)) * μ(z_face,h_ij) * ∂ₕbₘₗ
        
        if (z_face > -h_ij) & (npoints > minpoints)
            @inbounds Ψₑ[i, j, k] = min(Ψ_max, Ψ_ijk)
        else
            @inbounds Ψₑ[i, j, k] = 0.0
        end
    end

end

function compute_Ψₑ!(Ψₑ, h, ∂ₕb, ∂b∂z, μ, f; ce = 0.06, Lfₘ = 500meters, ΔS=10kilometers, τ=86400, minpoints=4, Vm=0.5)
    grid = h.grid
    arch = architecture(grid)


    event = launch!(arch, grid, :xy,
                    _compute_Ψₑ!, Ψₑ, grid, h, ∂ₕb, ∂b∂z, μ, f, ce, Lfₘ, ΔS, τ, minpoints, Vm,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)
    
    fill_halo_regions!(Ψₑ, arch)
    return nothing
end

# create field for each component of Ψ
Ψx = Field{Center, Face, Face}(grid)
Ψy = Field{Face, Center, Face}(grid)

# structure function
@inline μ(z,h) = (1-(2*z/h + 1)^2)*(1+(5/21)*(2*z/h + 1)^2)

# create a function for each component
compute_Ψx!(simulation) = compute_Ψₑ!(Ψx, @at((Center, Face, Nothing), h), ∂b∂y, ∂b∂z, μ, model.coriolis.f)
compute_Ψy!(simulation) = compute_Ψₑ!(Ψy, @at((Face, Center, Nothing), h), ∂b∂x, ∂b∂z, μ, model.coriolis.f)

# add the function to the callbacks of the simulation
simulation.callbacks[:compute_Ψx] = Callback(compute_Ψx!)
simulation.callbacks[:compute_Ψy] = Callback(compute_Ψy!)


# ----
# ---- compute mixed-layer eddy velocities

u_mle = Field(∂z(Ψy), data=u_mle.data)
v_mle = Field(∂z(Ψx), data=v_mle.data, boundary_conditions=v_mle_bc)
w_mle = Field(-(∂x(Ψy) + ∂y(Ψx)), data=w_mle.data, boundary_conditions=w_mle_bc)

function compute_mle_velocity!(sim)
    compute!(u_mle)
    compute!(v_mle)
    compute!(w_mle)
    return nothing
end


simulation.callbacks[:compute_mle_velocity] = Callback(compute_mle_velocity!)


outputs = merge(model.velocities, model.tracers, (; u_mle=u_mle, v_mle=v_mle, w_mle=w_mle, h, 
                                                    hx=@at((Center, Face, Nothing), h), hy=@at((Face, Center, Nothing), h), 
                                                    ∂b∂x, ∂b∂y, Ψx, Ψy)) # make a NamedTuple with all outputs

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filename = "../../data/raw/output_coarse_mle.nc",
                     schedule=TimeInterval(8hours))


using Printf

function print_progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour))

run!(simulation)