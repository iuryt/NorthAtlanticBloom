using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Architectures: device_event, architecture
using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Operators: Δzᶜᶜᶜ

@kernel function _cumulative_vertical_integration!(field, grid)
    i, j = @index(Global, NTuple)

    integ = 0
    @unroll for k in 1 : grid.Nz
        integ = integ + @inbounds field[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        @inbounds field[i, j, k] = integ
    end
end

function cumulative_vertical_integration!(input_field)
    field = Field(input_field)
    grid = field.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xy,
                    _cumulative_vertical_integration!, field, grid,
                    dependencies = device_event(arch))

    wait(device_event(arch), event)

    return field
end
