from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import acos, exp, pi, sin, sqrt
from typing import Dict, Iterable, Tuple

import numpy as np

G0 = 9.80665
Vec3 = np.ndarray

def _vec3(value: Vec3 | tuple[float, float, float]) -> Vec3:
    """Return an independent float vector and reject non-3D input."""
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError("expected a three-component vector")
    return result.copy()


def _norm(vector: Vec3) -> float:
    return float(np.linalg.norm(vector))


def _unit(vector: Vec3, fallback: Vec3 | None = None) -> Vec3:
    magnitude = _norm(vector)
    if magnitude > 1e-9:
        return vector / magnitude
    if fallback is not None:
        return _unit(fallback)
    raise ValueError("cannot normalize a zero-length vector")


def _angle_between(left: Vec3, right: Vec3) -> float:
    return acos(float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0)))

def _segment_miss(relative_start: Vec3, relative_end: Vec3) -> float:
    """Closest distance to the origin along a straight relative-motion segment."""
    delta = relative_end - relative_start
    span = float(np.dot(delta, delta))
    if span < 1e-12:
        return _norm(relative_end)
    fraction = float(np.clip(-np.dot(relative_start, delta) / span, 0.0, 1.0))
    return _norm(relative_start + fraction * delta)

def _turn_toward(current: Vec3, desired: Vec3, max_angle_rad: float) -> Vec3:
    """Turn a unit vector by no more than max_angle_rad in one time step."""
    current = _unit(current)
    desired = _unit(desired)
    angle = _angle_between(current, desired)

    if angle <= max_angle_rad or angle < 1e-9:
        return desired

    # Linear blend is sufficient for this game-grade attitude approximation.
    blend = max_angle_rad / angle
    return _unit((1.0 - blend) * current + blend * desired)


class StoreRole(str, Enum):
    AIR_TO_AIR = "air_to_air"

class StoreState(str, Enum):
    SAFE = "safe"
    ARMED = "armed"
    RELEASED = "released"

class FlightState(str, Enum):
    LAUNCHED = "launched"
    DETONATED = "detonated"
    EXPIRED = "expired"
    GROUND_IMPACT = "ground_impact"

@dataclass(frozen=True)
class Atmosphere:
    

    density_kg_m3: float
    speed_of_sound_mps: float

    @classmethod
    def isa(cls, altitude_m: float) -> "Atmosphere":
    
        altitude_m = max(0.0, altitude_m)
        gamma = 1.4
        gas_constant = 287.05287

        if altitude_m <= 11_000.0:
            temperature_k = 288.15 - 0.0065 * altitude_m
            pressure_pa = 101_325.0 * (temperature_k / 288.15) ** 5.25588
        else:
            temperature_k = 216.65
            pressure_11km_pa = 22_632.06
            pressure_pa = pressure_11km_pa * exp(
                -G0 * (altitude_m - 11_000.0) / (gas_constant * temperature_k)
            )

        return cls(
            density_kg_m3=pressure_pa / (gas_constant * temperature_k),
            speed_of_sound_mps=sqrt(gamma * gas_constant * temperature_k),
        )


@dataclass(frozen=True)
class launchSpec:
    """Intrinsic properties of a launch type, independent of the scenario."""

    name: str
    display_name: str
    tacview_type: str
    role: StoreRole

    launch_mass_kg: float
    propellant_mass_kg: float
    length_m: float
    diameter_m: float
    drag_coefficient: float

    motor_thrust_newton: float
    motor_burn_time_s: float

    speed_limit_mps: float
    max_lateral_load_g: float
    maximum_flight_time_s: float

    minimum_launch_speed_mps: float
    maximum_launch_speed_mps: float
    maximum_launch_altitude_m: float
    minimum_engagement_range_m: float
    maximum_engagement_range_m: float

    seeker_acquisition_range_m: float
    seeker_track_range_m: float
    seeker_acquisition_fov_rad: float
    seeker_track_fov_rad: float

    proximity_fuse_radius_m: float

    def __post_init__(self) -> None:
        positive = {
            "launch_mass_kg": self.launch_mass_kg,
            "length_m": self.length_m,
            "diameter_m": self.diameter_m,
            "drag_coefficient": self.drag_coefficient,
            "motor_thrust_newton": self.motor_thrust_newton,
            "motor_burn_time_s": self.motor_burn_time_s,
            "speed_limit_mps": self.speed_limit_mps,
            "max_lateral_load_g": self.max_lateral_load_g,
            "maximum_flight_time_s": self.maximum_flight_time_s,
            "maximum_launch_altitude_m": self.maximum_launch_altitude_m,
            "minimum_engagement_range_m": self.minimum_engagement_range_m,
            "maximum_engagement_range_m": self.maximum_engagement_range_m,
            "seeker_acquisition_range_m": self.seeker_acquisition_range_m,
            "seeker_track_range_m": self.seeker_track_range_m,
            "seeker_acquisition_fov_rad": self.seeker_acquisition_fov_rad,
            "seeker_track_fov_rad": self.seeker_track_fov_rad,
            "proximity_fuse_radius_m": self.proximity_fuse_radius_m,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        if not 0.0 < self.propellant_mass_kg < self.launch_mass_kg:
            raise ValueError("propellant mass must be between zero and launch mass")
        if not 0.0 <= self.minimum_launch_speed_mps < self.maximum_launch_speed_mps:
            raise ValueError("launch-speed limits must be non-negative and ordered")
        if self.maximum_launch_speed_mps > self.speed_limit_mps:
            raise ValueError("maximum launch speed cannot exceed speed limit")
        if self.minimum_engagement_range_m >= self.maximum_engagement_range_m:
            raise ValueError("engagement-range limits must be ordered")
        if self.seeker_acquisition_range_m > self.seeker_track_range_m:
            raise ValueError("track range must be at least acquisition range")
        if self.seeker_acquisition_fov_rad > self.seeker_track_fov_rad:
            raise ValueError("track field of view must be at least acquisition FOV")

    @property
    def burnout_mass_kg(self) -> float:
        return self.launch_mass_kg - self.propellant_mass_kg

    @property
    def reference_area_m2(self) -> float:
        return pi * (self.diameter_m / 2.0) ** 2

    @property
    def propellant_flow_kgps(self) -> float:
        return self.propellant_mass_kg / self.motor_burn_time_s


@dataclass(frozen=True)
class Hardpoint:
    """One F-16 attachment point. Coordinates are body-frame x/y/z in meters."""

    station_id: str
    max_store_mass_kg: float
    compatible_roles: Tuple[StoreRole, ...]
    body_position_m: Tuple[float, float, float]

    def accepts(self, launch: launchSpec) -> bool:
        return (
            launch.role in self.compatible_roles
            and launch.launch_mass_kg <= self.max_store_mass_kg
        )


@dataclass
class Mountedlaunch:
    spec: launchSpec
    hardpoint: Hardpoint
    state: StoreState = StoreState.SAFE

    def arm(self) -> None:
        if self.state is StoreState.RELEASED:
            raise RuntimeError("a released launch cannot be armed")
        self.state = StoreState.ARMED

    def safe(self) -> None:
        if self.state is StoreState.RELEASED:
            raise RuntimeError("a released launch cannot be safed")
        self.state = StoreState.SAFE

    def release(self) -> launchSpec:
        if self.state is not StoreState.ARMED:
            raise RuntimeError("only an armed launch may be released")
        self.state = StoreState.RELEASED
        return self.spec


@dataclass
class launchLoadout:
    """Captive stores belonging to one F-16 aircraft instance."""

    hardpoints: Iterable[Hardpoint]
    _stations: Dict[str, Hardpoint] = field(init=False, repr=False)
    _stores: Dict[str, Mountedlaunch] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.hardpoints = tuple(self.hardpoints)
        self._stations = {station.station_id: station for station in self.hardpoints}
        if not self._stations:
            raise ValueError("a loadout requires at least one hardpoint")
        if len(self._stations) != len(self.hardpoints):
            raise ValueError("hardpoint station IDs must be unique")

    def load(self, station_id: str, launch: launchSpec) -> Mountedlaunch:
        if station_id in self._stores:
            raise ValueError(f"hardpoint {station_id!r} is already occupied")

        if station_id not in self._stations:
            raise KeyError(f"unknown hardpoint {station_id!r}")

        hardpoint = self._stations[station_id]
        if not hardpoint.accepts(launch):
            raise ValueError(f"{launch.name} is incompatible with {station_id}")

        mounted = Mountedlaunch(spec=launch, hardpoint=hardpoint)
        self._stores[station_id] = mounted
        return mounted

    def get(self, station_id: str) -> Mountedlaunch | None:
        return self._stores.get(station_id)

    def release(self, station_id: str) -> launchSpec:
        mounted = self._stores.get(station_id)
        if mounted is None:
            raise KeyError(f"no launch loaded at {station_id!r}")

        spec = mounted.release()
        del self._stores[station_id]
        return spec


@dataclass(frozen=True)
class TargetState:
    identifier: str
    position_neu_m: Vec3
    velocity_neu_mps: Vec3
    alive: bool = True


@dataclass(frozen=True)
class LaunchConditions:
    """Information F16Env supplies when evaluating a launch attempt."""

    carrier_position_neu_m: Vec3
    carrier_velocity_neu_mps: Vec3
    carrier_nose_neu: Vec3
    target: TargetState
    target_track_valid: bool


@dataclass(frozen=True)
class LaunchAssessment:
    permitted: bool
    reason: str
    range_m: float
    boresight_rad: float


def evaluate_launch(spec: launchSpec, launch: LaunchConditions) -> LaunchAssessment:
    """Evaluate the instantaneous physical/seeker launch envelope."""
    own_speed = _norm(launch.carrier_velocity_neu_mps)
    altitude_m = float(launch.carrier_position_neu_m[2])
    line_of_sight = launch.target.position_neu_m - launch.carrier_position_neu_m
    distance_m = _norm(line_of_sight)
    boresight_rad = _angle_between(launch.carrier_nose_neu, line_of_sight)

    if not launch.target.alive:
        return LaunchAssessment(False, "target is not alive", distance_m, boresight_rad)
    if not launch.target_track_valid:
        return LaunchAssessment(False, "no valid target track", distance_m, boresight_rad)
    if not spec.minimum_launch_speed_mps <= own_speed <= spec.maximum_launch_speed_mps:
        return LaunchAssessment(False, "carrier speed outside launch limits", distance_m, boresight_rad)
    if altitude_m > spec.maximum_launch_altitude_m:
        return LaunchAssessment(False, "carrier altitude above launch limit", distance_m, boresight_rad)
    if not spec.minimum_engagement_range_m <= distance_m <= spec.maximum_engagement_range_m:
        return LaunchAssessment(False, "target outside engagement range", distance_m, boresight_rad)
    if distance_m > spec.seeker_acquisition_range_m:
        return LaunchAssessment(False, "target outside seeker acquisition range", distance_m, boresight_rad)
    if boresight_rad > spec.seeker_acquisition_fov_rad / 2.0:
        return LaunchAssessment(False, "target outside seeker acquisition field of view", distance_m, boresight_rad)

    return LaunchAssessment(True, "launch permitted", distance_m, boresight_rad)


@dataclass
class launchFlight:
    """Independent launch state after a store is released."""

    spec: launchSpec
    position_neu_m: Vec3
    velocity_neu_mps: Vec3
    target_id: str

    state: FlightState = FlightState.LAUNCHED
    time_since_launch_s: float = 0.0
    remaining_propellant_kg: float = 0.0
    seeker_locked: bool = False
    event: str | None = None

    @classmethod
    def launch(cls, spec: launchSpec, launch: LaunchConditions) -> "launchFlight":
        assessment = evaluate_launch(spec, launch)
        if not assessment.permitted:
            raise ValueError(f"launch rejected: {assessment.reason}")

        forward = _unit(launch.carrier_nose_neu)
        initial_position = (
            _vec3(launch.carrier_position_neu_m) + forward * (spec.length_m / 2.0)
        )
        initial_velocity = _vec3(launch.carrier_velocity_neu_mps) + forward * 12.0

        return cls(
            spec=spec,
            position_neu_m=initial_position,
            velocity_neu_mps=initial_velocity,
            target_id=launch.target.identifier,
            remaining_propellant_kg=spec.propellant_mass_kg,
            seeker_locked=True,
        )

    @property
    def mass_kg(self) -> float:
        return self.spec.burnout_mass_kg + self.remaining_propellant_kg

    @property
    def alive(self) -> bool:
        return self.state is FlightState.LAUNCHED

    def _update_seeker(self, target: TargetState) -> None:
        if not target.alive:
            self.seeker_locked = False
            return

        line_of_sight = target.position_neu_m - self.position_neu_m
        distance_m = _norm(line_of_sight)
        forward = _unit(self.velocity_neu_mps)

        if not self.seeker_locked:
            self.seeker_locked = (
                distance_m <= self.spec.seeker_acquisition_range_m
                and _angle_between(forward, line_of_sight)
                <= self.spec.seeker_acquisition_fov_rad / 2.0
            )
            return

        self.seeker_locked = (
            distance_m <= self.spec.seeker_track_range_m
            and _angle_between(forward, line_of_sight)
            <= self.spec.seeker_track_fov_rad / 2.0
        )

    def _guidance_direction(self, target: TargetState) -> Vec3:
        """Simple lead pursuit; replace later if a different game model is desired."""
        speed_mps = max(_norm(self.velocity_neu_mps), 1.0)
        line_of_sight = target.position_neu_m - self.position_neu_m
        time_to_go_s = _norm(line_of_sight) / speed_mps
        lead_position = target.position_neu_m + target.velocity_neu_mps * time_to_go_s
        return _unit(lead_position - self.position_neu_m)

    def step(self, dt_s: float, target: TargetState) -> None:
        """Advance the launch by one simulation interval."""
        if not self.alive:
            return
        if not 0.0 < dt_s <= 0.1:
            raise ValueError("launch step dt_s must be in (0, 0.1]")

        self.time_since_launch_s += dt_s
        if self.time_since_launch_s >= self.spec.maximum_flight_time_s:
            self.state = FlightState.EXPIRED
            self.event = "maximum flight time reached"
            return

        self._update_seeker(target)

        start_position_neu_m = self.position_neu_m.copy()
        speed_mps = max(_norm(self.velocity_neu_mps), 1.0)
        velocity_direction = _unit(self.velocity_neu_mps)
        gravity_neu_mps2 = np.array([0.0, 0.0, -G0])

        lateral_neu_mps2 = np.zeros(3)
        if self.seeker_locked:
            desired_velocity = self._guidance_direction(target) * speed_mps
            required = (desired_velocity - self.velocity_neu_mps) / dt_s - gravity_neu_mps2
            lateral_neu_mps2 = required - np.dot(required, velocity_direction) * velocity_direction
            lateral_limit = self.spec.max_lateral_load_g * G0
            lateral_mag = _norm(lateral_neu_mps2)
            if lateral_mag > lateral_limit:
                lateral_neu_mps2 *= lateral_limit / lateral_mag

        thrust_n = 0.0
        if self.remaining_propellant_kg > 0.0:
            burned_kg = min(self.remaining_propellant_kg, self.spec.propellant_flow_kgps * dt_s,)
            self.remaining_propellant_kg -= burned_kg
            thrust_n = self.spec.motor_thrust_newton

        thrust_n = 0.0
        if self.remaining_propellant_kg > 0.0:
            burned_kg = min(
                self.remaining_propellant_kg,
                self.spec.propellant_flow_kgps * dt_s,
            )
            self.remaining_propellant_kg -= burned_kg
            thrust_n = self.spec.motor_thrust_newton

        atmosphere = Atmosphere.isa(float(self.position_neu_m[2]))
        drag_n = (
            0.5
            * atmosphere.density_kg_m3
            * speed_mps**2
            * self.spec.drag_coefficient
            * self.spec.reference_area_m2
        )

        velocity_direction = _unit(self.velocity_neu_mps)
        acceleration_neu_mps2 = (
            velocity_direction * ((thrust_n - drag_n) / self.mass_kg)
            + lateral_neu_mps2
            + gravity_neu_mps2
        )

        self.velocity_neu_mps = self.velocity_neu_mps + acceleration_neu_mps2 * dt_s
        new_speed_mps = _norm(self.velocity_neu_mps)
        if new_speed_mps > self.spec.speed_limit_mps:
            self.velocity_neu_mps *= self.spec.speed_limit_mps / new_speed_mps

        self.position_neu_m = self.position_neu_m + self.velocity_neu_mps * dt_s

        if self.position_neu_m[2] <= 0.0:
            self.state = FlightState.GROUND_IMPACT
            self.event = "ground impact"
            return

        if target.alive:
            # closest approach during the interval, not just at its end point
            relative_end = target.position_neu_m - self.position_neu_m
            relative_start = relative_end - (
                target.velocity_neu_mps * dt_s - (self.position_neu_m - start_position_neu_m)
            )
            miss_distance_m = _segment_miss(relative_start, relative_end)
            if miss_distance_m <= self.spec.proximity_fuse_radius_m:
                self.state = FlightState.DETONATED
                self.event = f"proximity fuse at {miss_distance_m:.1f} m"

    def telemetry(self) -> dict[str, float | str | bool]:
        """Small record suitable for a later CSV/Tacview logging adapter."""
        return {
            "state": self.state.value,
            "time_since_launch_s": self.time_since_launch_s,
            "mass_kg": self.mass_kg,
            "speed_mps": _norm(self.velocity_neu_mps),
            "seeker_locked": self.seeker_locked,
            "position_n_m": float(self.position_neu_m[0]),
            "position_e_m": float(self.position_neu_m[1]),
            "position_up_m": float(self.position_neu_m[2]),
        }


shot9_STYLE_SHORT_RANGE_shotm = launchSpec(
    name="shotm-SR-01",
    display_name="shot-9-style launch",
    tacview_type="Weapon+launch+AirToAir",
    role=StoreRole.AIR_TO_AIR,
    launch_mass_kg=85.0,
    propellant_mass_kg=27.0,
    length_m=2.9,
    diameter_m=0.127,
    drag_coefficient=0.32,
    motor_thrust_newton=13_500.0,
    motor_burn_time_s=5.0,
    speed_limit_mps=780.0,
    max_lateral_load_g=30.0,
    maximum_flight_time_s=45.0,
    minimum_launch_speed_mps=120.0,
    maximum_launch_speed_mps=500.0,
    maximum_launch_altitude_m=18_000.0,
    minimum_engagement_range_m=400.0,
    maximum_engagement_range_m=18_000.0,
    seeker_acquisition_range_m=12_000.0,
    seeker_track_range_m=18_000.0,
    seeker_acquisition_fov_rad=np.radians(40.0),
    seeker_track_fov_rad=np.radians(70.0),
    proximity_fuse_radius_m=8.0,
)


# structural frame (x aft, z up) -> body frame (x fwd, y right, z down)
F16_WINGTIP_shotm_HARDPOINTS = (
    Hardpoint(
        station_id="left_wingtip",
        max_store_mass_kg=100.0,
        compatible_roles=(StoreRole.AIR_TO_AIR,),
        body_position_m=(-1.82, -4.80, -0.13),
    ),
    Hardpoint(
        station_id="right_wingtip",
        max_store_mass_kg=100.0,
        compatible_roles=(StoreRole.AIR_TO_AIR,),
        body_position_m=(-1.82, 4.80, -0.13),
    ),
)