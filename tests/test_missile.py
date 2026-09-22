import unittest

from testlaunch import (
    BASELINE_SHORT_RANGE_AAM,
    Hardpoint,
    MissileLoadout,
    StoreRole,
    StoreState,
)

class MissileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardpoint = Hardpoint(
            station_id="left_wingtip",
            max_store_mass_kg=100.0,
            compatible_roles=(StoreRole.AIR_TO_AIR,),
            body_position_m=(0.0, -4.8, 0.0),
        )

    def test_baseline_mass_and_motor_values_are_consistent(self) -> None:
        missile = BASELINE_SHORT_RANGE_AAM

        self.assertAlmostEqual(missile.burnout_mass_kg, 54.0)
        self.assertGreater(missile.reference_area_m2, 0.0)
        self.assertAlmostEqual(missile.total_impulse_newton_seconds, 94_350.0)

    def test_missile_can_be_loaded_armed_and_released(self) -> None:
        loadout = MissileLoadout((self.hardpoint,))
        mounted = loadout.load("left_wingtip", BASELINE_SHORT_RANGE_AAM)

        self.assertIs(mounted, loadout.get("left_wingtip"))
        mounted.arm()
        released_spec = loadout.release("left_wingtip")

        self.assertIs(released_spec, BASELINE_SHORT_RANGE_AAM)
        self.assertEqual(mounted.state, StoreState.RELEASED)
        self.assertIsNone(loadout.get("left_wingtip"))

if __name__ == "__main__":
    unittest.main()
