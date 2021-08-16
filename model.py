# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
from mpi4py import MPI as _MPI

comm = _MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import os
import numpy as np
import sys

# os.environ["UW_ENABLE_TIMING"] = "1"

import underworld as uw
import underworld.scaling as scaling

output_path = "../outputs/test_5/"

# %%
u = scaling.units
model_length = 1e4 * u.kilometer
model_top = 0.0 * u.kilometer
model_bottom = -2e3 * u.kilometer

refDensity = 3300.0 * u.kilogram / u.meter ** 3
refViscosity = 1e20 * u.pascal * u.second

KL = 2e3 * u.kilometer
KM = refDensity * KL ** 3
Kt = 1.0 / (refViscosity / KM * KL)

scaling_coefficients = scaling.get_coefficients()

scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients["[temperature]"] = 1400.0 * u.degK  # 放在 density 之前？

model_bottom = scaling.non_dimensionalise(model_bottom)
model_top = scaling.non_dimensionalise(model_top)
model_length = scaling.non_dimensionalise(model_length)

mesh = uw.mesh.FeMesh_Cartesian(
    elementRes=(1024, 256),
    minCoord=(-0.5 * model_length, model_bottom),
    maxCoord=(0.5 * model_length, model_top),
)

with mesh.deform_mesh():
    mesh.data[:, 0] = mesh.data[:, 0] * np.exp(
        (np.abs(mesh.data[:, 0]) - 0.5 * model_length) / 3.0
    )
    mesh.data[:, 1] = mesh.data[:, 1] * np.exp(
        (np.abs(mesh.data[:, 1]) - np.abs(model_bottom)) / 1.5
    )


# %%
trench_x = 0


def distance(p, c):
    return np.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)

PlateThickness = scaling.non_dimensionalise(100.0 * u.kilometer)
age_nondim_unit = scaling.non_dimensionalise(1 * u.second)
diffusivity = scaling.non_dimensionalise(1e-6 * u.metre ** 2 / u.second)

def plate_model(age, depth):
    """age unit : s; depth : non-dimensionlized"""
    age_nondim = age * age_nondim_unit
    T = depth / PlateThickness
    for i in range(1, 100):
        T = T + 2 / np.pi * (1 / i) * np.exp(
            -1 * (diffusivity * age_nondim * i ** 2 * np.pi ** 2 / PlateThickness ** 2)
        ) * np.sin(i * np.pi * depth / PlateThickness)
    return T


# %%
tempField = mesh.add_variable(1)
velocityField = mesh.add_variable(nodeDofCount=2)
pressureField = mesh.subMesh.add_variable(1)
velocityField.data[:] = (0.0, 0.0)
pressureField.data[:] = 0.0

# %% [markdown]
# scaling 过程不要出现在循环中，会很慢

# %%
radius = 300. / 2e3


# %%
adiabat_nondim = (2130 - 1573) / 1400
Temp = np.ones([1025, 257])

local_len = mesh.nodesDomain

for i in range(local_len):
    xv = mesh.data[i, 0]
    zv = mesh.data[i, 1]
    depth_nondim = np.abs(zv)
    tempField.data[i] = 1.0 + depth_nondim * adiabat_nondim
    if (
        depth_nondim < 100. / 2e3 # scaling.non_dimensionalise(100.0 * u.kilometer)
        and xv > trench_x
        and xv < 2.3
    ):
        xv_age_ma = (2.3 - xv) * 2e3 * 1e3 / 0.05 / 1e6
        tempField.data[i] = plate_model(xv_age_ma * 3.15e13, depth_nondim)
    if (
        depth_nondim < 100. / 2e3 # scaling.non_dimensionalise(100.0 * u.kilometer)
        and xv < trench_x
    ):
        xv_age_ma = (2.5 + xv) * 2e3 * 1e3 / 0.05 / 1e6
        tempField.data[i] = plate_model(xv_age_ma * 3.15e13, depth_nondim)
    if (
        xv >= trench_x
        and depth_nondim < 200. / 2e3 #scaling.non_dimensionalise(200.0 * u.kilometer)
        and distance(
            [xv, depth_nondim],
            [trench_x, radius],
        )
        > (radius - 100. / 2e3) # scaling.non_dimensionalise(100.0 * u.kilometer)
        and distance(
            [xv, depth_nondim],
            [trench_x, radius],
        )
        < (radius - 10. / 2e3) # scaling.non_dimensionalise(190.0 * u.kilometer)
    ):
        xv_age_ma = (2.5) * 2e3 * 1e3 / 0.05 / 1e6
        tempField.data[i] = plate_model(
            xv_age_ma * 3.15e13,
            radius
            - distance(
                [xv, depth_nondim],
                [trench_x, radius]
            )
        )

tempField.data[tempField.data < 0.0] = 0.0


# %%
# get_ipython().run_line_magic('matplotlib', 'widget')
# import matplotlib.pyplot as plt


# %%
# plt.plot([0,1])
# %matplotlib widget
# plt.scatter(mesh.data[:,0], mesh.data[:,1], c=tempField.data[:], s=0.01)


# %%
swarm = uw.swarm.Swarm(mesh=mesh)
swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(
    swarm=swarm, particlesPerCell=20
)

materialIndex = swarm.add_variable(dataType="int", count=1)
viscosityVar = swarm.add_variable(dataType="float", count=1)
stress = swarm.add_variable(dataType="float", count=3)


# %%
population_control = uw.swarm.PopulationControl(
    swarm,
    aggressive=True,
    splitThreshold=0.15,
    maxDeletions=2,
    maxSplits=10,
    particlesPerCell=20,
)

swarm.populate_using_layout(layout=swarmLayout)


# %%
for i in range(swarm.particleLocalCount):
    materialIndex.data[i] = 0
    xv = swarm.data[i, 0]
    zv = swarm.data[i, 1]
    depth_nondim = np.abs(zv)
    if (
        depth_nondim < 15/2e3
        and xv <= trench_x
    ):
        materialIndex.data[i] = 1
    if (
        xv >= trench_x
        and depth_nondim < 200. / 2e3
        and distance(
            [xv, depth_nondim],
            [trench_x, radius],
        )
        > (radius - 15 / 2e3) # scaling.non_dimensionalise(100.0 * u.kilometer)
        and distance(
            [xv, depth_nondim],
            [trench_x, radius],
        )
        < radius # scaling.non_dimensionalise(190.0 * u.kilometer)
    ):
        materialIndex.data[i] = 1


# %%
# %matplotlib widget
# plt.scatter(swarm.data[::200,0], swarm.data[::200,1], c=materialIndex.data[::200], s=0.01)

# %% [markdown]
# ## Material setting

# %%
crust = 1
mantle = 0


# %%
import underworld.function as fn

strainRateFn = fn.tensor.symmetric(velocityField.fn_gradient)
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)

# %% [markdown]
# #### diffusion

# %%
z_hat = uw.scaling.non_dimensionalise(model_top) - fn.input()[1]
P_stat = (
    uw.scaling.non_dimensionalise(refDensity)
    * uw.scaling.non_dimensionalise(10.0 * u.meter / u.second ** 2)
    * z_hat
)


# %%
E_diff = uw.scaling.non_dimensionalise(
    335.0 * 1e3 * u.joule / u.molar
)  # J/mol, activation energy
V_diff = uw.scaling.non_dimensionalise(
    4e-6 * u.meter ** 3 / u.molar
)  # m^3/mol, activation volume
A_diff = uw.scaling.non_dimensionalise(
    1.0e-9 / u.pascal / u.second
)  # Pa^-1.s^-1, prefactor
n_diff = 1.0


# %%
R = uw.scaling.non_dimensionalise(8.3145 * u.joule / u.molar / u.degK)
diffusionCreep = fn.math.pow(A_diff, (-1.0 / n_diff))
diffusionCreep *= fn.math.exp((E_diff + P_stat * V_diff) / (n_diff * R * (tempField+273.15/1400)))

# %% [markdown]
# #### dislocation

# %%
n_disl = 3.5
E_disl = uw.scaling.non_dimensionalise(
    480 * 1e3 * u.joule / u.molar
)  # J/mol, activation energy
V_disl = uw.scaling.non_dimensionalise(
    11e-6 * u.meter ** 3 / u.molar
)  # m^3/mol, activation volume
A_disl = uw.scaling.non_dimensionalise(
    3.11e-17 / u.pascal ** n_disl / u.second
)  # Pa^-n.s^-1, prefactor


# %%
dislocationCreep =  fn.math.pow(A_disl, (-1.0 / n_disl))
dislocationCreep *= fn.math.exp((E_disl + P_stat * V_disl) / (n_disl * R * (tempField+273.15/1400)))
dislocationCreep *= fn.math.pow(
    (strainRate_2ndInvariantFn + uw.scaling.non_dimensionalise(1.0e-24 / u.second)),
    (1 - n_disl) / n_disl,
)

# %% [markdown]
# #### stress limiter

# %%
yield_stress = uw.scaling.non_dimensionalise(5e8 * u.pascal)
reference_strainRate = uw.scaling.non_dimensionalise(1e-15 / u.second)
n_yield = 10


# %%
stress_limiter = fn.math.pow( strainRate_2ndInvariantFn + uw.scaling.non_dimensionalise(1.0e-24 / u.second) , 1/n_yield - 1) * yield_stress * fn.math.pow(reference_strainRate,-1/n_yield)


# %%
visc_UM = fn.math.pow((1.0 / diffusionCreep + 1.0 / dislocationCreep + 1.0 / stress_limiter) , -1)


# %%
# stress_limiter.evaluate([0.,-30/2e3])

# %% [markdown]
# #### lower mantle

# %%
E_diff_LM = uw.scaling.non_dimensionalise(
    2e5 * u.joule / u.molar
)  # J/mol, activation energy
V_diff_LM = uw.scaling.non_dimensionalise(
    1.1e-6 * u.meter ** 3 / u.molar
)  # m^3/mol, activation volume
A_diff_LM = uw.scaling.non_dimensionalise(
    5e-17 / u.pascal / u.second
)  # Pa^-1.s^-1, prefactor


# %%
visc_LM = fn.math.pow(A_diff_LM, -1.0)
visc_LM *= fn.math.exp((E_diff_LM + P_stat * V_diff_LM) / ( R * (tempField+273.15/1400)))

# %% [markdown]
# #### non-dimen

# %%
# visc_nondim = uw.scaling.dimensionalise(1, u.pascal * u.second)
# visc_nondim.magnitude


# %%
# visc_nondim = uw.scaling.dimensionalise(1, u.pascal * u.second)
# visc_UM = visc_UM / visc_nondim.magnitude
# visc_LM = visc_LM / visc_nondim.magnitude
# 不需要转换了

# %% [markdown]
# ### assign viscosity

# %%
visc_crust_value = uw.scaling.non_dimensionalise(1e20 * u.pascal * u.second)

viscMin = 1e-1
viscMax = 1e6

visc_mantle = fn.branching.conditional(
    [
        (
            z_hat < 660/2e3,
            fn.branching.conditional([
                (visc_UM > viscMax, viscMax),
                (visc_UM < viscMin, viscMin),
                (True, visc_UM)
            ])
        ),
        (
            True,
            fn.branching.conditional([
                (visc_LM > viscMax, viscMax),
                (visc_LM < viscMin, viscMin),
                (True, visc_LM)
            ])
        ),
    ]
)

visc_crust = fn.branching.conditional(
    [
        (
            z_hat < 400/2e3,
            visc_crust_value
        ),
        (True, visc_mantle),
    ]
)



mappingDictViscosity = {
    crust: visc_crust,
    mantle: visc_mantle
}
viscosityMapFn = fn.branching.map(fn_key=materialIndex, mapping=mappingDictViscosity)


# %%
# plt.scatter(swarm.data[:,0], swarm.data[:,1], c=viscosityMapFn.evaluate(swarm), s=0.01)
stress_fn = 2.0 * viscosityMapFn * strainRateFn

# %%
# import numpy as np
# plt.scatter(swarm.data[::100,0], swarm.data[::100,1], c=np.log10(viscosityMapFn.evaluate(swarm)[::100]), s=0.01)

# %% [markdown]
# ### end of visc
# 
gamma_410 = 3.0e6 * 1400 / 3300 / 10 / 2e6
temp_410 = 1.07462
delta_rho_410 = scaling.non_dimensionalise(273. * u.kilogram / u.meter ** 3)
width_410 = scaling.non_dimensionalise(13 * u.kilometer)
Phase_change_410_func = -1. * fn.input()[1] - 410/2e3 - gamma_410 * (tempField - temp_410)
Phase_change_410_density = delta_rho_410 * 0.5 * ( 1. + fn.math.tanh(Phase_change_410_func/width_410))

# %%
force_fn = (
    0.0,
    (uw.scaling.non_dimensionalise(refDensity)
    * (1.0 - uw.scaling.non_dimensionalise(3e-5 / u.degK) * tempField)
    + Phase_change_410_density 
    )
    * uw.scaling.non_dimensionalise(-10.0 * u.meter / u.second ** 2),
)


# %%
botSet = mesh.specialSets["Bottom_VertexSet"]
topSet = mesh.specialSets["Top_VertexSet"]
totSet = botSet + topSet
condition_temp = uw.conditions.DirichletCondition(tempField, totSet)


# %%
leftrightWalls = (
    mesh.specialSets["Left_VertexSet"] + mesh.specialSets["Right_VertexSet"]
)
topbottWalls = mesh.specialSets["Top_VertexSet"] + mesh.specialSets["Bottom_VertexSet"]
condition_vel = uw.conditions.DirichletCondition(
    velocityField, (leftrightWalls, topbottWalls)
)


# %%
stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    voronoi_swarm=swarm,
    conditions=condition_vel,
    fn_viscosity=viscosityMapFn,
    fn_bodyforce=force_fn,
)

solver = uw.systems.Solver(stokes)


# %%
solver.set_inner_method("superludist")
solver.set_penalty(1e6)


# %%
advector = uw.systems.SwarmAdvector(swarm=swarm, velocityField=velocityField, order=2)


# %%
advDiff = uw.systems.AdvectionDiffusion(
    phiField=tempField,
    method='SLCN',
    # phiDotField=temperatureDotField,
    velocityField=velocityField,
    fn_diffusivity=1.0 * uw.scaling.non_dimensionalise(1e-6 * u.metre ** 2 / u.second),
    conditions=condition_temp,
)


# %%
import sys

phase_410 = mesh.add_variable(1)


def savefiles(step,time):

    viscosityVar.data[:] = viscosityMapFn.evaluate(swarm)
    xdmf_info_swarm = swarm.save(output_path + "swarm_" + str(step) + ".h5")
    xdmf_info_vis = viscosityVar.save(output_path + "viscosity_" + str(step) + ".h5")
    xdmf_info_material = materialIndex.save(
        output_path + "material_" + str(step) + ".h5"
    )
    viscosityVar.xdmf(
        output_path + "swarm_visc_" + str(step) + ".xdmf",
        xdmf_info_vis,
        "viscosity",
        xdmf_info_swarm,
        "swarm",
        time
    )
    materialIndex.xdmf(
        output_path + "swarm_material_" + str(step) + ".xdmf",
        xdmf_info_material,
        "material",
        xdmf_info_swarm,
        "swarm",
        time
    )

    xdmf_info_mesh = mesh.save(output_path + "mesh_" + str(step) + ".h5")
    xdmf_info_temp = tempField.save(output_path + "temp_" + str(step) + ".h5")
    tempField.xdmf(
        output_path + "temp_" + str(step) + ".xdmf",
        xdmf_info_temp,
        "temperature",
        xdmf_info_mesh,
        "mesh",
        time
    )

    stress.data[:] = stress_fn.evaluate(swarm)
    xdmf_info_stress = stress.save(output_path + "stress_" + str(step) + ".h5")
    stress.xdmf(
        output_path + "stress_" + str(step) + ".xdmf",
        xdmf_info_stress,
        "stress",
        xdmf_info_swarm,
        "swarm",
        time
    )

    phase_410.data[:] = Phase_change_410_density.evaluate(mesh)
    xdmf_info_phase_410 = phase_410.save(output_path + "phase_410_" + str(step) + ".h5")
    phase_410.xdmf(
        output_path + "phase_410_" + str(step) + ".xdmf",
        xdmf_info_phase_410,
        "phase_410_density",
        xdmf_info_mesh,
        "mesh",
        time
    )

    xdmf_info_velo = velocityField.save(output_path + "velo_" + str(step) + ".h5")
    velocityField.xdmf(
        output_path + "velo_" + str(step) + ".xdmf",
        xdmf_info_velo,
        "velocity",
        xdmf_info_mesh,
        "mesh",
        time
    )

    # xdmf_info_submesh = mesh.subMesh.save(output_path+"submesh_"+str(step)+".h5")
    xdmf_info_pressure = pressureField.save(
        output_path + "pressure_" + str(step) + ".h5"
    )
    pressureField.xdmf(output_path+"pressure_"+str(step)+".xdmf",xdmf_info_pressure, "pressure", xdmf_info_mesh, "mesh",time)


# %%
time = 0.0
step = 0
steps_end = 1000
step_interval = 10


def update(time, step):
    # Retrieve the maximum possible timestep for the advection-diffusion system.
    # dt = np.min(advDiff.get_max_dt(), advector.get_max_dt())
    dt1 = advDiff.get_max_dt()
    dt2 = advector.get_max_dt()
    dt = np.min([dt1,dt2])
    # Advect using this timestep size.
    advDiff.integrate(dt)
    advector.integrate(dt)
    if step % step_interval == 0:
        savefiles(step, float(uw.scaling.dimensionalise(time, u.year) / u.year))
    population_control.repopulate()
    return time + dt, step + 1


# perform timestepping
while step < steps_end:
    # Solve for the velocity field given the current temperature field.
    solver.solve(nonLinearIterate=True)
    # velocityField.data[...] = 0.
    time, step = update(time, step)
    # print("done save file\n")
    # sys.stdout.flush()
    if uw.mpi.rank == 0:
        print(uw.scaling.dimensionalise(time, u.year), step)
    sys.stdout.flush()

