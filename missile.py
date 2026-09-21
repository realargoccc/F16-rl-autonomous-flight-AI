"""Foundational air-to-air missile store model for the F-16 environment.

All public values use SI units.  This module deliberately contains store
configuration and lifecycle only; propulsion, guidance, collision, and damage
are added in later simulation phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi
from typing import Dict, Iterable, Tuple


class StoreRole(str, Enum):
    """Mission role used when checking a store against a hardpoint."""

    AIR_TO_AIR = "air_to_air"


class StoreState(str, Enum):
    """Lifecycle states before a released missile becomes a flight object."""

    SAFE = "safe"
    ARMED = "armed"
    RELEASED = "released"


@dataclass(frozen=True)
class MissileSpec:
    """Immutable physical and operating limits for one missile type.

    ``motor_thrust_newton`` is an intentionally simple constant-thrust motor
    approximation.  It is sufficient for the first flight model and can later
    be replaced with a thrust-time curve without changing the store interface.
    """

    name: str
    role: StoreRole
    launch_mass_kg: float
    propellant_mass_kg: float
    length_m: float
    diameter_m: float
    motor_thrust_newton: float
    motor_burn_time_s: float
    speed_limit_mps: float
    max_lateral_load_g: float
    minimum_launch_speed_mps: float
    maximum_launch_speed_mps: float
    maximum_launch_altitude_m: float

    def __post_init__(self) -> None:
        positive_values = {
            "launch_mass_kg": self.launch_mass_kg,
            "length_m": self.length_m,
            "diameter_m": self.diameter_m,
            "motor_thrust_newton": self.motor_thrust_newton,
            "motor_burn_time_s": self.motor_burn_time_s,
            "speed_limit_mps": self.speed_limit_mps,
            "max_lateral_load_g": self.max_lateral_load_g,
            "maximum_launch_altitude_m": self.maximum_launch_altitude_m,
        }
        for field_name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")

        if not 0.0 < self.propellant_mass_kg < self.launch_mass_kg:
            raise ValueError("propellant_mass_kg must be between zero and launch mass")
        if not 0.0 <= self.minimum_launch_speed_mps < self.maximum_launch_speed_mps:
            raise ValueError("launch-speed limits must be non-negative and ordered")
        if self.maximum_launch_speed_mps > self.speed_limit_mps:
            raise ValueError("maximum launch speed cannot exceed the missile speed limit")

    @property
    def burnout_mass_kg(self) -> float:
        """Mass remaining after all motor propellant has been consumed."""

        return self.launch_mass_kg - self.propellant_mass_kg

    @property
    def reference_area_m2(self) -> float:
        """Frontal reference area for the later drag calculation."""

        return pi * (self.diameter_m / 2.0) ** 2

    @property
    def total_impulse_newton_seconds(self) -> float:
        """Constant-thrust motor impulse used for first-pass sanity checks."""

        return self.motor_thrust_newton * self.motor_burn_time_s


@dataclass(frozen=True)
class Hardpoint:
    """An F-16 station that can accept a compatible missile store.

    Body coordinates follow the future adapter's convention: x forward,
    y right, z down, in meters.  They are metadata until JSBSim mass and drag
    integration is enabled.
    """

    station_id: str
    max_store_mass_kg: float
    compatible_roles: Tuple[StoreRole, ...]
    body_position_m: Tuple[float, float, float]

    def accepts(self, missile: MissileSpec) -> bool:
        return (
            missile.role in self.compatible_roles
            and missile.launch_mass_kg <= self.max_store_mass_kg
        )


@dataclass
class MountedMissile:
    """A missile loaded on one hardpoint before its independent flight begins."""

    spec: MissileSpec
    hardpoint: Hardpoint
    state: StoreState = StoreState.SAFE

    def arm(self) -> None:
        if self.state is StoreState.RELEASED:
            raise RuntimeError("a released missile cannot be armed")
        self.state = StoreState.ARMED

    def safe(self) -> None:
        if self.state is StoreState.RELEASED:
            raise RuntimeError("a released missile cannot be safed")
        self.state = StoreState.SAFE

    def release(self) -> MissileSpec:
        """Detach an armed store and return the spec for the flight model."""

        if self.state is not StoreState.ARMED:
            raise RuntimeError("only an armed missile may be released")
        self.state = StoreState.RELEASED
        return self.spec


@dataclass
class MissileLoadout:
    """Owns the missile stores fitted to a single aircraft instance."""

    hardpoints: Iterable[Hardpoint]
    _stations: Dict[str, Hardpoint] = field(init=False, repr=False)
    _stores: Dict[str, MountedMissile] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.hardpoints = tuple(self.hardpoints)
        self._stations = {station.station_id: station for station in self.hardpoints}
        if not self._stations:
            raise ValueError("a loadout requires at least one hardpoint")
        if len(self._stations) != len(self.hardpoints):
            raise ValueError("hardpoint station IDs must be unique")

    def load(self, station_id: str, missile: MissileSpec) -> MountedMissile:
        if station_id in self._stores:
            raise ValueError(f"hardpoint {station_id!r} is already occupied")
        try:
            hardpoint = self._stations[station_id]
        except KeyError as exc:
            raise KeyError(f"unknown hardpoint {station_id!r}") from exc
        if not hardpoint.accepts(missile):
            raise ValueError(f"{missile.name} is incompatible with {station_id}")

        mounted = MountedMissile(spec=missile, hardpoint=hardpoint)
        self._stores[station_id] = mounted
        return mounted

    def get(self, station_id: str) -> MountedMissile | None:
        return self._stores.get(station_id)

    def release(self, station_id: str) -> MissileSpec:
        try:
            mounted = self._stores[station_id]
        except KeyError as exc:
            raise KeyError(f"no missile loaded at {station_id!r}") from exc
        spec = mounted.release()
        del self._stores[station_id]
        return spec


# A deliberately generic, short-range air-to-air baseline.  These values are
# plausible for an F-16-compatible store, not a claim to model a named weapon.
BASELINE_SHORT_RANGE_AAM = MissileSpec(
    name="AAM-SR-01",
    role=StoreRole.AIR_TO_AIR,
    launch_mass_kg=86.0,
    propellant_mass_kg=32.0,
    length_m=2.87,
    diameter_m=0.127,
    motor_thrust_newton=18_500.0,
    motor_burn_time_s=5.1,
    speed_limit_mps=750.0,
    max_lateral_load_g=30.0,
    minimum_launch_speed_mps=120.0,
    maximum_launch_speed_mps=500.0,
    maximum_launch_altitude_m=18_000.0,
)
