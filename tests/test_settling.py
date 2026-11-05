from dataclasses import replace
import numpy as np
import warp as wp
from test_transport import BuildTransport, MeanTemperature
from config import TransportConfig


def BuildResting(history=False):
    sim = BuildTransport(TransportConfig(enabled=True, settling_duration_s=.0004), history=history, gas=True)
    sim.simulation_config = replace(sim.simulation_config, gravity=(0.,-9.81,0.))
    sim.material.simulation_config = sim.simulation_config
    return sim


def TestRestIsIsothermalMassConservingAndFallsThenHeats():
    sim = BuildResting(history=True)
    initial = sim.current_state
    water = initial.transport_state.water_masses.numpy().copy()
    gas = initial.material_state.gas_moles.numpy().copy()
    y = initial.particle_positions.numpy()[:,1].mean()
    sim.Advance(4)
    settled = sim.current_state
    np.testing.assert_array_equal(settled.transport_state.water_masses.numpy(), water)
    np.testing.assert_array_equal(settled.material_state.gas_moles.numpy(), gas)
    np.testing.assert_allclose(settled.transport_state.temperatures.numpy(),293.15,atol=1e-12)
    np.testing.assert_array_equal(settled.transport_state.water_losses.numpy(),0.)
    np.testing.assert_array_equal(settled.transport_state.heat_exchanges.numpy(),0.)
    assert settled.particle_positions.numpy()[:,1].mean() < y
    assert settled.particle_velocities.numpy()[:,1].mean() < 0
    velocity = settled.particle_velocities.numpy().copy()
    sim.Step()
    assert sim.baking_time == sim.simulation_config.time_step
    assert sim.current_state.transport_state.temperatures.numpy().max() > 293.15
    assert sim.current_state.material_state.gas_moles.numpy().min() > gas.min()
    assert sim.current_state.particle_velocities.numpy()[:,1].mean() < velocity[:,1].mean()
    assert len(sim.state_history) == 6 and sim.state_history[4] is settled


def EvaluateRestGradient(transfer, backward=False):
    sim=BuildResting(history=True)
    sim.transport.heat_transfer_parameter.assign([transfer])
    loss=wp.zeros(1,dtype=wp.float64,device='cpu',requires_grad=True)
    with wp.Tape() as tape:
        for _ in range(8):
            sim.Step()
        wp.launch(MeanTemperature,sim.particle_data.particle_count,inputs=[sim.current_state.transport_state.temperatures,loss],device='cpu')
    if backward:
        tape.backward(loss)
    return loss.numpy()[0], float(sim.transport.heat_transfer_parameter.grad.numpy()[0]) if backward else None


def TestGradientAcrossRestBakingBoundary():
    _,automatic=EvaluateRestGradient(20.,True)
    finite=(EvaluateRestGradient(22.)[0]-EvaluateRestGradient(18.)[0])/4
    assert automatic > 0
    np.testing.assert_allclose(automatic,finite,rtol=.003,atol=1e-10)


def TestZeroRestPreservesLegacyAndDurationRounding():
    sim=BuildResting()
    sim.transport.config=replace(sim.transport.config,settling_duration_s=.00041)
    assert sim.settling_steps == 5
    sim.transport.config=replace(sim.transport.config,settling_duration_s=0.)
    assert sim.settling_steps == 0
    sim.Step()
    assert sim.current_state.transport_state.temperatures.numpy().max() > 293.15
