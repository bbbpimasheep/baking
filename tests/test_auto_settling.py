from dataclasses import replace
import numpy as np
import pytest
import warp as wp
from test_transport import BuildTransport, MeanTemperature
from config import TransportConfig


def BuildAuto(history=False, gravity=(0.,0.,0.)):
    config=TransportConfig(enabled=True,settling_duration_s=.002,settling_auto=True,
        settling_min_duration_s=.0002,settling_hold_duration_s=.0003,
        settling_speed_tolerance=1e-4,settling_rate_tolerance=.01)
    sim=BuildTransport(config,history=history,gas=True)
    sim.simulation_config=replace(sim.simulation_config,gravity=gravity)
    sim.material.simulation_config=sim.simulation_config
    return sim


def TestQuietConfirmationThenBakingAndReset(tmp_path):
    from experiments.recording import SimulationRecorder
    from config import ExperimentConfig,RunConfig
    sim=BuildAuto(history=True)
    config=ExperimentConfig(material=sim.material.material_config,simulation=sim.simulation_config,
        ding=sim.material.ding_config,transport=sim.transport.config,run=RunConfig(log_path=str(tmp_path/'results')))
    recorder=SimulationRecorder(sim,config)
    sim.result_recorder=recorder
    initial=sim.current_state
    for _ in range(4):
        sim.Step()
        assert sim.IsSettling and sim.baking_time==0
        np.testing.assert_allclose(sim.current_state.transport_state.temperatures.numpy(),293.15,atol=1e-12)
    sim.Step()
    assert not sim.IsSettling and sim.settling_reason=='quasi_static'
    assert sim.settling_steps==5 and sim.baking_time==0
    settled=sim.current_state
    assert recorder.settled_measurement is not None and recorder.pending_snapshots
    assert not (tmp_path/'results').exists()
    sim.Step()
    assert sim.baking_steps==1 and sim.baking_time==sim.simulation_config.time_step
    assert sim.current_state.transport_state.temperatures.numpy().max()>293.15
    assert sim.state_history[0] is initial and sim.state_history[5] is settled
    path=recorder.Finish()
    import json
    summary=json.loads((path/'summary.json').read_text())
    assert summary['settling_reason']=='quasi_static' and summary['settling_steps']==5
    assert (path/'snapshot_0000005.npz').exists()
    sim.Reset()
    assert sim.IsSettling and sim.settling_quiet_steps==0 and sim.baking_steps==0


def TestMovingObjectTimesOutWithoutHeat():
    sim=BuildAuto(gravity=(0.,-9.81,0.))
    with pytest.raises(RuntimeError,match='baking was not started'):
        sim.Advance(20)
    assert sim.IsSettling and sim.settling_reason=='timeout' and sim.baking_time==0
    np.testing.assert_allclose(sim.current_state.transport_state.temperatures.numpy(),293.15,atol=1e-12)


def TestBriefQuietAndDeformationCannotTriggerTransition():
    sim=BuildAuto()
    sim.Advance(3)
    assert sim.settling_quiet_steps==1
    sim.current_state.particle_velocities.assign(np.tile([.01,0.,0.],(sim.particle_data.particle_count,1)))
    sim.Step()
    assert sim.IsSettling and sim.settling_quiet_steps==0
    from physics.quasistatic import SettlingMetrics
    count=1
    old=wp.array([np.eye(3)],dtype=wp.mat33,device='cpu')
    changed=wp.array([np.eye(3)*1.01],dtype=wp.mat33,device='cpu')
    metrics=wp.zeros(3,dtype=float,device='cpu')
    wp.launch(SettlingMetrics,1,inputs=[wp.zeros(1,dtype=wp.vec3,device='cpu'),
        wp.array([np.eye(3)*.1],dtype=wp.mat33,device='cpu'),old,changed,metrics,.01],device='cpu')
    values=metrics.numpy()
    assert values[0]==0 and values[1]>.1 and values[2]>.9


def EvaluateAutoGradient(value,backward=False):
    sim=BuildAuto(history=True)
    sim.transport.heat_transfer_parameter.assign([value])
    loss=wp.zeros(1,dtype=wp.float64,device='cpu',requires_grad=True)
    with wp.Tape() as tape:
        for _ in range(8):
            sim.Step()
        wp.launch(MeanTemperature,sim.particle_data.particle_count,inputs=[sim.current_state.transport_state.temperatures,loss],device='cpu')
    if backward:
        tape.backward(loss)
    assert sim.settling_steps==5
    return loss.numpy()[0],float(sim.transport.heat_transfer_parameter.grad.numpy()[0]) if backward else None


def TestGradientWithUnchangedStoppingStep():
    _,automatic=EvaluateAutoGradient(20.,True)
    finite=(EvaluateAutoGradient(22.)[0]-EvaluateAutoGradient(18.)[0])/4
    np.testing.assert_allclose(automatic,finite,rtol=.003,atol=1e-10)
