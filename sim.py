import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, SPSA
from qiskit.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

seed = 69
mapper = JordanWignerMapper()
driver = PySCFDriver(
    atom=f"H 0 0 0; H 0 0 0.725",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

def create_ansatz(es_problem):
    return UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

def create_vqe_solver(estimator, ansatz):
    vqe_solver = VQE(estimator, ansatz, SPSA())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    return vqe_solver

class ThermalNoiseModel(NoiseModel):
    def __init__(self, t1, t2):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.time_cx = 300
        self.time_reset = 1000
        self.time_measure = 1000

        self.add_errors()

    def add_errors(self):
        self.add_all_qubit_quantum_error(thermal_relaxation_error(self.t1, self.t2, self.time_reset), "reset")
        self.add_all_qubit_quantum_error(thermal_relaxation_error(self.t1, self.t2, self.time_measure), "measure")
        self.add_all_qubit_quantum_error(
            thermal_relaxation_error(self.t1, self.t2, self.time_cx).expand(thermal_relaxation_error(self.t1, self.t2, self.time_cx)), "cx")


class EnergyCalculator:
    def __init__(self, acc):
        self.estimator = AerEstimator(
            backend_options={
                "method": "density_matrix",
                "noise_model": ThermalNoiseModel(
                    acc * 10e3,
                    acc * 15e3
                ),
            },
            run_options={"seed": seed, "shots": 8192},
            transpile_options={"seed_transpiler": seed},
        )
        self.calc = GroundStateEigensolver(mapper, create_vqe_solver(self.estimator, ansatz))
        self.vqe_energies = []

    def calculate(self):
        self.vqe_energies = []
        for distance in distances:
            driver = PySCFDriver(
                atom=f"H 0 0 0; H 0 0 {distance}",
                basis="sto3g",
                charge=0,
                spin=0,
                unit=DistanceUnit.ANGSTROM,
            )
            es_problem = driver.run()
            res = self.calc.solve(es_problem)
            self.vqe_energies.append(res.eigenvalues[0] + res._nuclear_repulsion_energy)

    def minimum(self):
        interpolated_function = interp1d(distances, self.vqe_energies, kind='cubic')
        result = minimize_scalar(interpolated_function, bounds=(distances.min(), distances.max()), method='bounded')
        return result.fun, result.x

# es_problem = driver.run()
# ansatz = create_ansatz(es_problem)
distances = np.hstack((np.arange(0.2, 1.55, 0.05), np.arange(1.75, 4.25, 0.25)))
#
# # accuracies = np.arange(1, 21, 1)
# # results = []
# # for accuracy in tqdm(accuracies, desc="Evaluating simulation"):
# #     calculator = EnergyCalculator(accuracy)
# #     calculator.calculate()
# #     results.append(calculator.minimum())
#
# calculator = EnergyCalculator(150)
# calculator.calculate()
# print(calculator.minimum())

import multiprocessing
def calculate_energy(distance):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    es_problem = driver.run()
    ansatz = create_ansatz(es_problem)
    vqe_solver = create_vqe_solver(AerEstimator(
            backend_options={
                "method": "density_matrix",
                "noise_model": ThermalNoiseModel(
                    100e3,
                    150e3
                ),
            },
            run_options={"seed": seed, "shots": 8192},
            transpile_options={"seed_transpiler": seed},
        ), ansatz)
    qubit_op, aux_op = es_problem.second_q_ops()
    qubit_op, aux_op = mapper.map(qubit_op), mapper.map(aux_op)
    raw_result = vqe_solver.compute_minimum_eigenvalue(qubit_op, aux_op)
    res = es_problem.interpret(raw_result)
    return res.total_energies[0]

if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        vqe_energies = list(tqdm(pool.imap_unordered(calculate_energy, distances), total=len(distances), desc="Calculating energies"))
    print(vqe_energies)

# noisy_estimator = AerEstimator(
#     backend_options={
#         "method": "density_matrix",
#         "noise_model": ThermalNoiseModel(),
#     },
#     run_options={"seed": seed, "shots": 1024},
#     transpile_options={"seed_transpiler": seed},
# )
#
# calc = GroundStateEigensolver(mapper, create_vqe_solver(create_ansatz(es_problem)))
# vqe_energies = []
#
# for distance in tqdm(distances, desc="Calculating energies"):
#     driver = PySCFDriver(
#         atom=f"H 0 0 0; H 0 0 {distance}",
#         basis="sto3g",
#         charge=0,
#         spin=0,
#         unit=DistanceUnit.ANGSTROM,
#     )
#     es_problem = driver.run()
#     res = calc.solve(es_problem)
#     vqe_energies.append(res.eigenvalues[0] + res._nuclear_repulsion_energy)
#
# print(vqe_energies)
#
# interpolated_function = interp1d(distances, vqe_energies, kind='cubic')
# result = minimize_scalar(interpolated_function, bounds=(distances.min(), distances.max()), method='bounded')
#
# min_distance = result.x
# min_energy = result.fun
# print(f"The distance that minimizes the energy is: {min_distance} Å")
# print(f"The minimum energy is: {min_energy} Hartree")

# fine_distances = np.linspace(distances.min(), distances.max(), 500)
# interpolated_energies = interpolated_function(fine_distances)
# plt.figure(figsize=(15, 10))
# plt.style.use('fast')
# plt.plot(fine_distances, interpolated_energies, label='Interpolated Energy')
# plt.plot(min_distance, min_energy, 'go', label='Minimum Energy')
# plt.plot(distances, vqe_energies, marker='o', color='red', linestyle='', markersize=5, label='VQE Energy')
# plt.xlabel('Atomic Distance (Å)')
# plt.ylabel('Energy (Hartree)')
# plt.title('Energy vs Atomic Distance for H2 Molecule')
# plt.grid(True)
# plt.legend()
# plt.show()