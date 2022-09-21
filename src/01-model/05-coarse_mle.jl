using Bioceananigans
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!

sinking = true
mle = true

#--------------- Grid

using NCDatasets
ds = Dataset("../../data/interim/input_coarse.nc")

i = 97 # index for time=12days

const initial_time = ds["time"][i]
const H = 1000 # maximum depth


const Nx, Ny, Nz = size(ds["b"][:,:,:,i])

grid = RectilinearGrid(GPU(),
    size=(Nx, Ny, Nz),
    x=(minimum(ds["xF"]),maximum(ds["xF"])+diff(ds["xF"])[end]),
    y=(minimum(ds["yF"]),maximum(ds["yF"])+diff(ds["yF"])[end]),
    z=ds["zF"][:],
    topology=(Periodic, Bounded, Bounded)
)


#--------------- Coriolis

coriolis = FPlane(latitude=60)

#--------------- Bottom drag

const cᴰ = 1e-4 # quadratic drag coefficient

@inline bottom_drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline bottom_drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

bottom_drag_bc_u = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
bottom_drag_bc_v = FluxBoundaryCondition(bottom_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = bottom_drag_bc_v)


#--------------- Sponges

const sponge_size = 50kilometers
const slope = 10kilometers
const ymin = minimum(ynodes(Center, grid))
const ymax = maximum(ynodes(Center, grid))

@inline mask_func(x,y,z) = ((
     tanh((y-(ymax-sponge_size))/slope)
    *tanh((y-(ymin+sponge_size))/slope)
)+1)/2

mom_sponge = Relaxation(rate=1/1hour, mask=mask_func, target=0)

horizontal_closure = HorizontalScalarDiffusivity(ν=10, κ=10)
vertical_closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

#--------------- NP Model

# constants for the NP model
const μ₀ = 1.25/day   # surface growth rate
const m = 0.015/day # mortality rate due to virus and zooplankton grazing
const Kw = 0.059 # meter^-1
const kn = 0.75
const kr = 0.5

#  https://doi.org/10.1029/2017GB005850
const chl2c = 0.06 # average value for winter in North Atlantic

const α = 0.0538/day

const average = :growth
const shading = true

# create the mld field that will be updated at every timestep
h = Field{Center, Center, Nothing}(grid) 
light_growth = Field{Center, Center, Center}(grid)


# time evolution of shortwave radiation (North Atlantic)
@inline Lₒ(t) = 116 * sin( 2π * ( t / days + 50 ) / 375.3 - 1.3 ) + 132.3
# evolution of the available light at the surface
@inline light_function(t, z) = 0.43 * Lₒ(t) * exp( z * Kw )
# light profile
@inline light_growth_function(light) = μ₀ * ( light * α ) / sqrt( μ₀^2 + ( light * α )^2 )


# nitrate and ammonium limiting
@inline N_lim(N, Nr) = (N/(N+kn)) * (kr/(Nr+kr))
@inline Nr_lim(Nr) =  (Nr/(Nr+kr))

# functions for the NP model
@inline P_forcing(light_growth, P, N, Nr)  =   light_growth * (N_lim(N, Nr) + Nr_lim(Nr)) * P - m * P^2
@inline N_forcing(light_growth, P, N, Nr)  = - light_growth * N_lim(N, Nr) * P
@inline Nr_forcing(light_growth, P, N, Nr) = - light_growth * Nr_lim(Nr) * P + m * P^2

# functions for the NP model
@inline P_forcing(i, j, k, grid, clock, fields, p)  = @inbounds P_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])
@inline N_forcing(i, j, k, grid, clock, fields, p)  = @inbounds N_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])
@inline Nr_forcing(i, j, k, grid, clock, fields, p) = @inbounds Nr_forcing(p.light_growth[i, j, k], fields.P[i, j, k], fields.N[i, j, k], fields.Nr[i, j, k])

# using the functions to determine the forcing
P_dynamics = Forcing(P_forcing, discrete_form=true, parameters=(; light_growth))
N_dynamics = Forcing(N_forcing, discrete_form=true, parameters=(; light_growth))
Nr_dynamics = Forcing(Nr_forcing, discrete_form=true, parameters=(; light_growth))

# sinking velocity

# Vertical velocity function
const w_sink = ifelse(sinking,-1meter/day,0)
const lamb = 1meters
@inline w_func(x, y, z) = w_sink * tanh(max(-z / lamb, 0.0)) * tanh(max((z + H) / lamb, 0.0))

no_penetration = ImpenetrableBoundaryCondition()
w_bc = FieldBoundaryConditions(grid, (Center, Center, Face), top=no_penetration, bottom=no_penetration)

# Field (allocates memory and precalculates w_func)
w = Field((Center, Center, Face), grid, boundary_conditions=w_bc)
set!(w, w_func)
P_sink = AdvectiveForcing(WENO5(; grid), w = w)


#--------------- Initial Mixed-layer Eddy Velocities

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

#--------------- Instantiate Model

if mle
    forcing = (;
        P=(P_dynamics, P_sink, mle_forcing), N=(N_dynamics, mle_forcing), 
        Nr=(Nr_dynamics, mle_forcing), b=mle_forcing,
        u=mom_sponge, v=mom_sponge, w=mom_sponge,
    )
else
    forcing = (;
        P=(P_dynamics, P_sink), N=(N_dynamics), 
        Nr=(Nr_dynamics),
        u=mom_sponge, v=mom_sponge, w=mom_sponge,
    )
end

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            coriolis = coriolis,
                            closure=(horizontal_closure,vertical_closure),
                            tracers = (:b, :P, :N, :Nr),
                            buoyancy = BuoyancyTracer(),
                            forcing = forcing,
                            boundary_conditions = (u=u_bcs, v=v_bcs))

model.clock.time = initial_time*days

bᵢ = Float64.(ds["b"][:,:,:,i])
Pᵢ = Float64.(ds["P"][:,:,:,i])
Nᵢ = Float64.(ds["N"][:,:,:,i])
Nrᵢ = Float64.(ds["Nr"][:,:,:,i])

set!(model; b=bᵢ, P=Pᵢ, N=Nᵢ, Nr=Nrᵢ)


#--------------- Initial Geostrophic Velocities

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

#--------------- Simulation

simulation = Simulation(model, Δt = 1minutes, stop_time = 80day)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1hour)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


#--------------- Mixed Layer Depth
const g = 9.82
const ρₒ = 1026

h = Field{Center, Center, Nothing}(grid) 
# buoyancy decrease criterium for determining the mixed-layer depth
const Δb = (g/ρₒ) * 0.03
compute_mixed_layer_depth!(simulation) = MixedLayerDepth!(h, simulation.model.tracers.b, Δb)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_mld] = Callback(compute_mixed_layer_depth!)

#--------------- Light-limiting growth

compute_light_growth!(simulation) = LightGrowth!(light_growth, h, simulation.model.tracers.P, light_function, light_growth_function, time(simulation), average, shading, chl2c)
# add the function to the callbacks of the simulation
simulation.callbacks[:compute_light_growth] = Callback(compute_light_growth!)

#--------------- Zeroing negative values

# zeroing negative values
@inline function zeroing(sim)
    @unroll for tracer in [:P, :N, :Nr]
        parent(sim.model.tracers[tracer]) .= max.(0, parent(sim.model.tracers[tracer]))
    end
end
simulation.callbacks[:zeroing] = Callback(zeroing)

#--------------- Compute gradient of buoyancy

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

#--------------- Compute Ψₑ

# create field for each component of Ψ
Ψx = Field{Center, Face, Face}(grid)
Ψy = Field{Face, Center, Face}(grid)

# structure function
@inline μ(z,h) = (1-(2*z/h + 1)^2)*(1+(5/21)*(2*z/h + 1)^2)

# import functions for mle_parameterization
include("mle_parameterization.jl")

# create a function for each component
compute_Ψx!(simulation) = compute_Ψₑ!(Ψx, @at((Center, Face, Nothing), h), ∂b∂y, ∂b∂z, μ, model.coriolis.f)
compute_Ψy!(simulation) = compute_Ψₑ!(Ψy, @at((Face, Center, Nothing), h), ∂b∂x, ∂b∂z, μ, model.coriolis.f)

# add the function to the callbacks of the simulation
simulation.callbacks[:compute_Ψx] = Callback(compute_Ψx!)
simulation.callbacks[:compute_Ψy] = Callback(compute_Ψy!)


#--------------- Compute mixed-layer eddy velocities

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


#--------------- Writing Outputs

bi, Pi, Ni, Nri = simulation.model.tracers
u, v, w = simulation.model.velocities


extra_outputs = (; 
    h=h, 
    light_growth=light_growth, 
    new_production=light_growth * N_lim(Ni, Nri) * Pi,
    u=@at((Center, Center, Center), u),
    v=@at((Center, Center, Center), v),
    w=@at((Center, Center, Center), w),
    N2=@at((Center, Center, Center), ∂z(bi)),
    ∇b=@at((Center, Center, Center), sqrt(∂x(bi)^2 + ∂y(bi)^2)),
    Ro=@at((Center, Center, Center), (∂x(v)-∂y(u))/f),
    u_mle=u_mle, v_mle=v_mle, w_mle=w_mle, 
    hx=@at((Center, Face, Nothing), h), 
    hy=@at((Face, Center, Nothing), h), 
    ∂b∂x, ∂b∂y, Ψx, Ψy
)



if mle
    filename = "../../data/raw/output_coarse_mle"
else
    filename = "../../data/raw/output_coarse_no_mle"
end

if sinking
    filename = filename*"_mu$(μ₀*days).nc"
else
    filename = filename*"_nosinking_mu$(μ₀*days).nc"
end



simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, merge(model.tracers, extra_outputs), 
                    filename = filename,
                    overwrite_existing=true,
                    schedule=TimeInterval(8hours))


#--------------- Printing Progress

using Printf

function print_progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, Ψmax = (%.1e, %.1e), wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(abs, Ψx), maximum(abs, Ψy),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(12hour))

run!(simulation)
