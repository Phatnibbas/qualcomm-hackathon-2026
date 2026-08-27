"""Target contract: exact formula vectors and fail-closed input handling."""

from __future__ import annotations

import math
import unittest

import numpy as np

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.target import (
    TargetInputError,
    apparent_temperature_array,
    apparent_temperature_c,
    target_contract,
    vapour_pressure_hpa,
)


def longhand_vapour_pressure(ta: float, rh: float) -> float:
    """The cited equation, written out independently of the module."""
    return (rh / 100.0) * 6.105 * math.exp(17.27 * ta / (237.7 + ta))


def longhand_apparent_temperature(ta: float, rh: float, ws: float) -> float:
    return ta + 0.33 * longhand_vapour_pressure(ta, rh) - 0.70 * ws - 4.00


class TestExactFormula(unittest.TestCase):
    """The implementation must reproduce the BoM equation exactly."""

    VECTORS = [
        (30.0, 70.0, 2.0, 29.60101300453148, 34.368334291495394),
        (25.0, 50.0, 0.0, 15.791481260295475, 26.211188815897508),
        (35.5, 85.0, 4.25, 48.9440474977948, 44.67653567427228),
    ]

    def test_vapour_pressure_matches_fixed_vectors(self):
        for ta, rh, _ws, expected_e, _expected_at in self.VECTORS:
            with self.subTest(ta=ta, rh=rh):
                self.assertAlmostEqual(vapour_pressure_hpa(ta, rh), expected_e, places=12)

    def test_apparent_temperature_matches_fixed_vectors(self):
        for ta, rh, ws, _expected_e, expected_at in self.VECTORS:
            with self.subTest(ta=ta, rh=rh, ws=ws):
                self.assertAlmostEqual(apparent_temperature_c(ta, rh, ws), expected_at, places=12)

    def test_matches_independent_longhand_equation(self):
        for ta in (18.0, 27.3, 34.9):
            for rh in (35.0, 62.5, 99.0):
                for ws in (0.0, 1.1, 6.0):
                    with self.subTest(ta=ta, rh=rh, ws=ws):
                        self.assertAlmostEqual(
                            apparent_temperature_c(ta, rh, ws),
                            longhand_apparent_temperature(ta, rh, ws),
                            places=12,
                        )

    def test_no_wind_height_correction_is_applied(self):
        """A height correction would break the exact wind coefficient."""
        base = apparent_temperature_c(30.0, 70.0, 0.0)
        with_wind = apparent_temperature_c(30.0, 70.0, 1.0)
        self.assertAlmostEqual(base - with_wind, 0.70, places=12)


class TestInvalidInputs(unittest.TestCase):
    """Every rejection below must raise, not degrade."""

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(TargetInputError):
                    apparent_temperature_c(bad, 70.0, 1.0)
                with self.assertRaises(TargetInputError):
                    apparent_temperature_c(30.0, bad, 1.0)
                with self.assertRaises(TargetInputError):
                    apparent_temperature_c(30.0, 70.0, bad)

    def test_rejects_relative_humidity_outside_range(self):
        for bad in (-0.1, 100.1, 250.0):
            with self.subTest(bad=bad):
                with self.assertRaises(TargetInputError):
                    apparent_temperature_c(30.0, bad, 1.0)

    def test_rejects_negative_wind(self):
        with self.assertRaises(TargetInputError):
            apparent_temperature_c(30.0, 70.0, -0.01)

    def test_rejects_relative_humidity_supplied_as_a_fraction(self):
        """0.7 is a fraction, not 0.7 %RH. Silently accepting it is the bug."""
        guard = float(load_config()["target"]["rh_fraction_guard_max_percent"])
        for bad in (0.35, 0.7, guard):
            with self.subTest(bad=bad):
                with self.assertRaises(TargetInputError):
                    apparent_temperature_c(30.0, bad, 1.0)

    def test_accepts_the_boundary_just_above_the_guard(self):
        guard = float(load_config()["target"]["rh_fraction_guard_max_percent"])
        self.assertTrue(math.isfinite(apparent_temperature_c(30.0, guard + 0.5, 1.0)))

    def test_zero_humidity_is_admitted_by_the_formula(self):
        """RH = 0 is physically meaningful; QC rejects it separately."""
        self.assertAlmostEqual(apparent_temperature_c(30.0, 0.0, 0.0), 26.0, places=12)


class TestVectorised(unittest.TestCase):
    def test_array_form_agrees_with_scalar_form(self):
        temp = np.array([30.0, 25.0, 35.5])
        rh = np.array([70.0, 50.0, 85.0])
        wind = np.array([2.0, 0.0, 4.25])
        produced = apparent_temperature_array(temp, rh, wind)
        for index in range(3):
            self.assertAlmostEqual(
                float(produced[index]),
                apparent_temperature_c(float(temp[index]), float(rh[index]), float(wind[index])),
                places=12,
            )

    def test_array_form_returns_nan_for_invalid_elements(self):
        produced = apparent_temperature_array(
            np.array([30.0, 30.0, 30.0, float("nan")]),
            np.array([70.0, 120.0, 0.7, 70.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        )
        self.assertTrue(math.isfinite(float(produced[0])))
        self.assertTrue(np.isnan(produced[1]))
        self.assertTrue(np.isnan(produced[2]), "RH given as a fraction must not produce a value")
        self.assertTrue(np.isnan(produced[3]))


class TestContract(unittest.TestCase):
    def test_contract_states_the_wind_height_transfer_limit(self):
        contract = target_contract()
        wind = contract["wind_height"]
        self.assertEqual(wind["station_sensor_height_m_agl"], 15.0)
        self.assertEqual(wind["bom_reference_wind_height_m"], 10.0)
        self.assertFalse(wind["wind_height_correction_applied"])
        self.assertIn("no site-validated", wind["why_no_correction"].lower())

    def test_contract_names_what_the_target_is_not(self):
        forbidden = " ".join(target_contract()["claim_boundary"]["is_not"]).lower()
        for phrase in ("wbgt", "legal", "medical", "safe/unsafe", "direct-sun"):
            self.assertIn(phrase, forbidden)

    def test_contract_does_not_fabricate_a_calibrated_uncertainty(self):
        uncertainty = target_contract()["uncertainty"]
        self.assertIsNone(uncertainty["calibrated_uncertainty_degc"])
        self.assertIn("unknown", uncertainty["reason"].lower())

    def test_contract_constants_match_the_configuration(self):
        config = load_config()
        contract = target_contract(config)
        self.assertEqual(
            contract["constants"]["vapour_pressure"], config["target"]["vapour_pressure_constants"]
        )
        self.assertEqual(
            contract["constants"]["apparent_temperature"],
            config["target"]["apparent_temperature_coefficients"],
        )


if __name__ == "__main__":
    unittest.main()
