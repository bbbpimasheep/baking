from dataclasses import replace
import json
import numpy as np
import pytest
import warp as wp
from config import DingConfig, MaterialConfig, SimulationConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.state import AllocateState
from physics.materials.ding import DingMaterial
from physics.mpm import MpmSimulator


def MakePoint(material_config=None, ding_config=None, requires_grad=False):
    wp.init()
    material = DingMaterial(material_config or MaterialConfig(), ding_config or DingConfig(), 'cpu', requires_grad)
    state = AllocateState(1, 'cpu', requires_grad)
    state.deformation_gradients.assign(np.eye(3, dtype=np.float32)[None])
    material.InitializeState(state)
    return material, state


def PointStep(material, old_state, total_gradient, time_step=0.01):
    output_state = AllocateState(1, 'cpu', old_state.particle_positions.requires_grad)
    increment = total_gradient @ np.linalg.inv(old_state.deformation_gradients.numpy()[0])
    output_state.deformation_gradients.assign(total_gradient.astype(np.float32)[None])
    output_state.affine_velocities.assign(((increment-np.eye(3))/time_step).astype(np.float32)[None])
    material.AdvanceState(old_state, output_state, time_step)
    return output_state


@pytest.mark.parametrize('stretch_value', [1.25, 0.8])
def TestPointLoadHoldUnload(stretch_value):
    material, state = MakePoint(ding_config=DingConfig(yield_stress=500.0, viscosity=1e10))
    total_gradient = np.diag([stretch_value, 1.0, 1.0])
    state = PointStep(material, state, total_gradient)
    initial_stress = state.material_state.kirchhoff_stresses.numpy()[0].copy()
    energies = []
    for step_index in range(40):
        state = PointStep(material, state, total_gradient)
        elastic_gradient = state.material_state.elastic_gradients.numpy()[0]
        strains = np.log(np.linalg.svd(elastic_gradient, compute_uv=False))
        energies.append(material.material_config.shear_modulus*np.dot(strains,strains) +
                        0.5*material.material_config.lame_lambda*strains.sum()**2)
    final_stress = state.material_state.kirchhoff_stresses.numpy()[0]
    assert initial_stress[0,0] * (stretch_value-1) > 0
    assert np.linalg.norm(final_stress-np.eye(3)*np.trace(final_stress)/3) < np.linalg.norm(initial_stress-np.eye(3)*np.trace(initial_stress)/3)
    assert np.max(np.diff(energies)) < 0.01
    np.testing.assert_allclose(np.linalg.det(state.material_state.plastic_gradients.numpy()[0]), 1.0, atol=2e-5)
    np.testing.assert_allclose(state.material_state.elastic_gradients.numpy()[0] @ state.material_state.plastic_gradients.numpy()[0], total_gradient, atol=1e-5)
    assert state.material_state.return_residuals.numpy().max() < 1e-5
    # Unload elastically to zero stress: Fnew = Fp, rather than forcing Fnew=I.
    residual_gradient = state.material_state.plastic_gradients.numpy()[0].copy()
    state = PointStep(material, state, residual_gradient)
    np.testing.assert_allclose(state.material_state.kirchhoff_stresses.numpy(), 0, atol=0.5)
    assert np.linalg.norm(residual_gradient-np.eye(3)) > 0.01


def TestElasticLimitObjectivityAndPorosity():
    material, state = MakePoint(ding_config=DingConfig(yield_stress=1e9))
    total_gradient = np.diag([1.1, 0.95, 1.02])
    state = PointStep(material, state, total_gradient)
    strains = np.log(np.diag(total_gradient))
    expected_stress = np.diag(2*material.material_config.shear_modulus*strains + material.material_config.lame_lambda*strains.sum())
    np.testing.assert_allclose(state.material_state.kirchhoff_stresses.numpy()[0], expected_stress, rtol=2e-5, atol=0.02)
    np.testing.assert_allclose(state.material_state.plastic_gradients.numpy()[0], np.eye(3), atol=1e-6)
    rotation_angle = 0.5
    rotation_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle),0], [np.sin(rotation_angle),np.cos(rotation_angle),0], [0,0,1]])
    state = PointStep(material, state, rotation_matrix @ total_gradient)
    np.testing.assert_allclose(state.material_state.kirchhoff_stresses.numpy()[0], rotation_matrix@expected_stress@rotation_matrix.T, atol=0.05)
    for porosity in (0.0,0.2):
        material, state = MakePoint(ding_config=DingConfig(yield_stress=100.0,porosity=porosity))
        state = PointStep(material,state, np.eye(3)*0.9)
        plastic_volume = np.linalg.det(state.material_state.plastic_gradients.numpy()[0])
        if porosity == 0:
            np.testing.assert_allclose(plastic_volume,1,atol=1e-5)
        else:
            assert plastic_volume < 0.999


@wp.kernel
def StressLoss(stresses: wp.array(dtype=wp.mat33), loss_value: wp.array(dtype=float)):
    particle_index = wp.tid()
    wp.atomic_add(loss_value,0, stresses[particle_index][0,0] / float(stresses.shape[0]))


def EvaluateGradient(viscosity_value, backward=False):
    material, state = MakePoint(ding_config=DingConfig(yield_stress=500.0,viscosity=viscosity_value), requires_grad=True)
    loss_value = wp.zeros(1,dtype=float,device='cpu',requires_grad=True)
    tape_context = wp.Tape()
    history_states = [state]
    with tape_context:
        for step_index in range(4):
            total_gradient = np.diag([1.15+0.01*step_index,0.93,1.02])
            state = PointStep(material,state,total_gradient)
            history_states.append(state)
        wp.launch(StressLoss,dim=1,inputs=[state.material_state.kirchhoff_stresses,loss_value],device='cpu')
    gradient_value = None
    if backward:
        tape_context.backward(loss_value)
        gradient_value = float(material.viscosity_parameter.grad.numpy()[0])
    return float(loss_value.numpy()[0]),gradient_value


def TestViscosityGradient():
    _, gradient_value = EvaluateGradient(1e10,True)
    epsilon_value = 1e8
    upper_value,_ = EvaluateGradient(1e10+epsilon_value)
    lower_value,_ = EvaluateGradient(1e10-epsilon_value)
    finite_difference = (upper_value-lower_value)/(2*epsilon_value)
    assert abs(finite_difference)>1e-12
    np.testing.assert_allclose(gradient_value,finite_difference,rtol=0.02,atol=1e-12)
    print('eta AD:',gradient_value,'FD:',finite_difference)


@pytest.mark.parametrize('device_name',['cpu','cuda:0'])
def TestMpmAndInternalSnapshots(tmp_path,device_name):
    wp.init()
    if device_name.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    tet_mesh = CreateDemoMesh(2)
    simulator = MpmSimulator(tet_mesh,SampleParticles(tet_mesh,1000.0,4),
        replace(SimulationConfig(),device_name=device_name),MaterialConfig(model='ding2019'),ding_config=DingConfig(yield_stress=100.0,viscosity=1e8))
    positions = simulator.current_state.particle_positions.numpy().copy()
    strain_rates = np.diag([1.5,-0.75,0.2]).astype(np.float32)
    simulator.current_state.particle_velocities.assign((positions-np.array([0.5,0.6,0.5],dtype=np.float32))@strain_rates.T)
    simulator.current_state.affine_velocities.assign(np.tile(strain_rates,(len(positions),1,1)))
    simulator.Advance(600)
    material_state = simulator.current_state.material_state
    assert np.max(np.abs(material_state.plastic_gradients.numpy()-np.eye(3))) > 1e-4
    np.testing.assert_allclose(material_state.elastic_gradients.numpy()@material_state.plastic_gradients.numpy(),simulator.current_state.deformation_gradients.numpy(),atol=2e-5)
    simulator.SaveSnapshot(tmp_path/'ding.npz')
    with np.load(tmp_path/'ding.npz') as snapshot_data:
        np.testing.assert_array_equal(snapshot_data['ding_elastic_gradients'],material_state.elastic_gradients.numpy())
        np.testing.assert_array_equal(snapshot_data['ding_plastic_gradients'],material_state.plastic_gradients.numpy())
    simulator.Reset()
    np.testing.assert_allclose(simulator.current_state.material_state.elastic_gradients.numpy(),np.tile(np.eye(3),(simulator.particle_data.particle_count,1,1)))


def EvaluateMpmGradient(viscosity_value, backward=False):
    tet_mesh = CreateDemoMesh(1)
    simulator = MpmSimulator(tet_mesh,SampleParticles(tet_mesh,1000.0,4),
        replace(SimulationConfig(),device_name='cpu',time_step=0.001,gravity=(0.0,0.0,0.0)),
        MaterialConfig(model='ding2019'),retain_history=True,
        ding_config=DingConfig(yield_stress=5.0,viscosity=viscosity_value))
    positions = simulator.current_state.particle_positions.numpy().copy()
    strain_rates = np.diag([1.5,-0.75,0.2]).astype(np.float32)
    simulator.current_state.particle_velocities.assign((positions-np.array([0.5,0.6,0.5],dtype=np.float32))@strain_rates.T)
    simulator.current_state.affine_velocities.assign(np.tile(strain_rates,(len(positions),1,1)))
    loss_value=wp.zeros(1,dtype=float,device='cpu',requires_grad=True)
    tape_context=wp.Tape()
    with tape_context:
        for step_index in range(5):
            simulator.Step()
        wp.launch(StressLoss,dim=len(positions),inputs=[simulator.current_state.material_state.kirchhoff_stresses,loss_value],device='cpu')
    simulator.CheckState()
    gradient_value=None
    if backward:
        tape_context.backward(loss_value)
        gradient_value=float(simulator.material.viscosity_parameter.grad.numpy()[0])
    return float(loss_value.numpy()[0]),gradient_value


def TestMpmViscosityGradient():
    _,gradient_value=EvaluateMpmGradient(1000.0,True)
    upper_value,_=EvaluateMpmGradient(1010.0)
    lower_value,_=EvaluateMpmGradient(990.0)
    finite_difference=(upper_value-lower_value)/20.0
    assert abs(finite_difference)>1e-7
    print('MPM eta AD:',gradient_value,'FD:',finite_difference)
    np.testing.assert_allclose(gradient_value,finite_difference,rtol=0.03,atol=1e-7)


def TestLocalStationarityAndViscosityDependence():
    norms=[]
    for viscosity_value in (1e9,1e11):
        material,state=MakePoint(ding_config=DingConfig(yield_stress=300.0,viscosity=viscosity_value,porosity=0.2))
        total_gradient=np.diag([1.2,0.9,1.04])
        state=PointStep(material,state,total_gradient)
        tau=np.diag(state.material_state.kirchhoff_stresses.numpy()[0]).astype(float)
        mu=material.material_config.shear_modulus
        lam=material.material_config.lame_lambda
        compliance=np.linalg.inv(2*mu*np.eye(3)+lam*np.ones((3,3)))
        trial_tau=(2*mu*np.eye(3)+lam*np.ones((3,3)))@np.log(np.diag(total_gradient))
        mean=tau.mean()
        dev=tau-mean
        a=material.pressure_coefficient
        f=1.5*np.dot(dev,dev)+a*mean**2-300.0**2
        grad_f=3*dev+(2*a/3)*mean*np.ones(3)
        stationarity=compliance@(tau-trial_tau)+0.01/viscosity_value*max(f,0)*grad_f
        np.testing.assert_allclose(stationarity,0,atol=2e-6)
        norms.append(np.linalg.norm(dev))
    assert norms[1]>norms[0]


def TestLoadRateDependence():
    stresses=[]
    for total_time in (0.02,0.2):
        material,state=MakePoint(ding_config=DingConfig(yield_stress=500.0,viscosity=1e10))
        for step_index in range(20):
            stretch_value=np.exp(np.log(1.2)*(step_index+1)/20)
            state=PointStep(material,state,np.diag([stretch_value,1.0,1.0]),total_time/20)
        tau=state.material_state.kirchhoff_stresses.numpy()[0]
        stresses.append(np.linalg.norm(tau-np.trace(tau)/3*np.eye(3)))
    assert stresses[0]>stresses[1]
