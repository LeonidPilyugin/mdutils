import numpy as np
import ase
import openmm as mm

class Cell:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_ase(cell):
        return cell * 0.1 * mm.unit.nanometer

class Positions:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_ase(positions):
        return positions * 0.1 * mm.unit.nanometer

    @staticmethod
    def from_openmm(positions):
        return positions.value_in_unit(mm.unit.nanometer) * 10.0

class Velocities:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_ase(velocities):
        return (
            velocities
            * 100.0
            * ase.units.fs
            * mm.unit.nanometer
            / mm.unit.picosecond
        )

    @staticmethod
    def from_openmm(velocities):
        return velocities.value_in_unit(
            mm.unit.nanometer / mm.unit.picosecond
        ) / 100.0 / ase.units.fs

class Energy:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_openmm(energy):
        return energy.value_in_unit(
            mm.unit.kilojoule_per_mole
        ) * ase.units.kJ / ase.units.mol

class Forces:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_openmm(energy):
        return energy.value_in_unit(
            mm.unit.kilojoule_per_mole / mm.unit.nanometer
        ) * ase.units.kJ / ase.units.mol / 10.0

class Masses:
    def __init__(self):
        assert not "Reached"

    @staticmethod
    def from_ase(masses):
        return masses * mm.unit.amu
