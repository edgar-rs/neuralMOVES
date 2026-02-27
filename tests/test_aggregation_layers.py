"""Tests for NeuralMOVES aggregation layers.

Tests cover:
- Default data loading and validation
- Drive cycle creation and evaluation
- Fleet composition from MOVES defaults
- Fleet-average emission rates
- Temporal aggregation (VMT, annualization)
- Technology scenarios (EV ramps)
- Geography (county profiles)
- Scenario comparison
- Key invariants (e.g., more EVs → less fleet CO2)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ── Defaults ──────────────────────────────────────────────────────────────────

class TestDefaults:
    """Test default data loading and validation."""

    def test_load_age_distribution(self):
        from neuralmoves.defaults import load_age_distribution
        df = load_age_distribution()
        assert len(df) > 30000
        assert set(df.columns) == {"sourceTypeID", "yearID", "ageID", "ageFraction"}

    def test_age_distribution_sums_to_one(self):
        from neuralmoves.defaults import load_age_distribution
        df = load_age_distribution()
        # Check a specific (sourceType, year) group
        group = df[(df["sourceTypeID"] == 21) & (df["yearID"] == 2019)]
        assert abs(group["ageFraction"].sum() - 1.0) < 0.01

    def test_load_month_vmt_fraction(self):
        from neuralmoves.defaults import load_month_vmt_fraction
        df = load_month_vmt_fraction()
        assert len(df) > 100
        assert "monthVMTFraction" in df.columns

    def test_month_fractions_sum_to_one(self):
        from neuralmoves.defaults import load_month_vmt_fraction
        df = load_month_vmt_fraction()
        for st_id, group in df.groupby("sourceTypeID"):
            total = group["monthVMTFraction"].sum()
            assert abs(total - 1.0) < 0.01, f"sourceType {st_id}: sum={total}"

    def test_load_sample_vehicle_population(self):
        from neuralmoves.defaults import load_sample_vehicle_population
        df = load_sample_vehicle_population()
        assert len(df) > 10000
        assert "stmyFraction" in df.columns

    def test_load_hpms_vtype_year(self):
        from neuralmoves.defaults import load_hpms_vtype_year
        df = load_hpms_vtype_year()
        assert len(df) > 200
        assert "HPMSBaseYearVMT" in df.columns

    def test_load_county_meteorology(self):
        from neuralmoves.defaults import load_county_meteorology
        df = load_county_meteorology()
        assert len(df) > 30000
        assert "temperature_F" in df.columns
        assert "relHumidity_pct" in df.columns

    def test_caching(self):
        from neuralmoves.defaults import load_age_distribution
        df1 = load_age_distribution()
        df2 = load_age_distribution()
        assert df1 is df2  # Same object (cached)


# ── Drive Cycles ──────────────────────────────────────────────────────────────

class TestDriveCycle:
    """Test drive cycle creation and properties."""

    def test_constant_speed(self):
        from neuralmoves.cycles import DriveCycle
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=600)
        assert cycle.name == "Constant 30 mph"
        assert cycle.duration_s == 599.0
        assert cycle.distance_miles > 0
        assert cycle.avg_speed_mph > 0
        assert len(cycle.speed_ms) == 600

    def test_accel_derivation(self):
        from neuralmoves.cycles import DriveCycle
        cycle = DriveCycle.constant_speed(speed_mph=60, duration_s=100)
        accel = cycle.accel_mps2
        assert len(accel) == len(cycle.speed_ms)
        # Middle of constant section should have ~0 acceleration
        assert abs(accel[50]) < 0.1

    def test_from_speed_array(self):
        from neuralmoves.cycles import DriveCycle
        speeds = np.array([0, 10, 20, 30, 30, 30, 20, 10, 0], dtype=float)
        cycle = DriveCycle.from_speed_array(speeds, dt=1.0, name="test")
        assert cycle.name == "test"
        assert len(cycle.speed_ms) == 9

    def test_default_grade(self):
        from neuralmoves.cycles import DriveCycle
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=100)
        assert np.all(cycle.grade_pct == 0)


class TestCycleEvaluation:
    """Test cycle evaluation with actual NN inference."""

    def test_evaluate_cycle(self):
        from neuralmoves.cycles import DriveCycle, evaluate_cycle
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=60)
        result = evaluate_cycle(
            cycle, temp=25.0, humid_pct=50.0,
            model_year=2019, source_type="Passenger Car", fuel_type="Gasoline",
        )
        assert result.total_co2_g > 0
        assert result.rate_g_per_mile > 0
        assert result.rate_g_per_km > 0
        assert len(result.timeseries_g_per_s) == 60

    def test_rate_reasonable_range(self):
        """Passenger car at 30 mph should be 100-500 g/mi."""
        from neuralmoves.cycles import DriveCycle, evaluate_cycle
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=300)
        result = evaluate_cycle(
            cycle, temp=25.0, humid_pct=50.0,
            model_year=2019, source_type="Passenger Car", fuel_type="Gasoline",
        )
        assert 100 < result.rate_g_per_mile < 500

    def test_higher_speed_different_rate(self):
        from neuralmoves.cycles import DriveCycle, evaluate_cycle
        cycle_30 = DriveCycle.constant_speed(speed_mph=30, duration_s=300)
        cycle_60 = DriveCycle.constant_speed(speed_mph=60, duration_s=300)

        r30 = evaluate_cycle(
            cycle_30, temp=25.0, humid_pct=50.0,
            model_year=2019, source_type="Passenger Car", fuel_type="Gasoline",
        )
        r60 = evaluate_cycle(
            cycle_60, temp=25.0, humid_pct=50.0,
            model_year=2019, source_type="Passenger Car", fuel_type="Gasoline",
        )
        # Different speeds should give different rates
        assert r30.rate_g_per_mile != r60.rate_g_per_mile


# ── Fleet ─────────────────────────────────────────────────────────────────────

class TestFleetComposition:
    """Test fleet composition building and manipulation."""

    def test_from_moves_defaults(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
        assert fleet.n_vehicle_configs > 0
        assert fleet.calendar_year == 2019

    def test_fractions_sum(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
        total = fleet.fractions["fraction"].sum() + fleet.ev_penetration
        assert abs(total - 1.0) < 0.01

    def test_with_ev_penetration(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
        fleet_ev = fleet.with_ev_penetration(0.30)
        assert fleet_ev.ev_penetration == 0.30
        ice_total = fleet_ev.fractions["fraction"].sum()
        assert abs(ice_total - 0.70) < 0.01

    def test_ev_zero(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(
            calendar_year=2019, ev_penetration=0.0,
        )
        assert fleet.ev_penetration == 0.0
        assert abs(fleet.fractions["fraction"].sum() - 1.0) < 0.01

    def test_ev_one(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(
            calendar_year=2019, ev_penetration=0.0,
        )
        fleet_all_ev = fleet.with_ev_penetration(1.0)
        assert fleet_all_ev.ev_penetration == 1.0
        assert fleet_all_ev.fractions["fraction"].sum() < 0.001

    def test_summary(self):
        from neuralmoves.fleet import FleetComposition
        fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
        summary = fleet.summary()
        assert len(summary) > 0
        assert "source_type_name" in summary.columns


class TestFleetAverage:
    """Test fleet-average emission rate computation."""

    def test_fleet_average_rate(self):
        from neuralmoves.cycles import DriveCycle
        from neuralmoves.fleet import FleetComposition, fleet_average_rate
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=60)
        fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
        result = fleet_average_rate(
            cycle, fleet, temp=25.0, humid_pct=50.0,
        )
        assert result.fleet_avg_g_per_mile > 0
        assert len(result.by_config) > 0

    def test_more_ev_less_co2(self):
        """Key invariant: higher EV penetration → lower fleet CO2."""
        from neuralmoves.cycles import DriveCycle
        from neuralmoves.fleet import FleetComposition, fleet_average_rate

        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=60)
        base = FleetComposition.from_moves_defaults(
            calendar_year=2019, ev_penetration=0.0,
        )

        r0 = fleet_average_rate(
            cycle, base.with_ev_penetration(0.0),
            temp=25.0, humid_pct=50.0,
        )
        r50 = fleet_average_rate(
            cycle, base.with_ev_penetration(0.50),
            temp=25.0, humid_pct=50.0,
        )
        # 50% EV should give ~50% reduction
        assert r50.fleet_avg_g_per_mile < r0.fleet_avg_g_per_mile
        ratio = r50.fleet_avg_g_per_mile / r0.fleet_avg_g_per_mile
        assert 0.4 < ratio < 0.6  # ~50% reduction


# ── Temporal ──────────────────────────────────────────────────────────────────

class TestTemporal:
    """Test VMT allocation and annualization."""

    def test_vmt_allocation(self):
        from neuralmoves.temporal import VMTAllocation
        vmt = VMTAllocation.from_moves_defaults(
            source_type_id=21, calendar_year=2019,
        )
        assert vmt.annual_vmt > 0
        assert len(vmt.month_fractions) == 12

    def test_monthly_vmt_sums(self):
        from neuralmoves.temporal import VMTAllocation
        vmt = VMTAllocation.from_moves_defaults(
            source_type_id=21, calendar_year=2019,
        )
        monthly = vmt.monthly_vmt()
        assert abs(monthly["monthly_vmt"].sum() - vmt.annual_vmt) < 1.0

    def test_annualize_emissions(self):
        from neuralmoves.temporal import VMTAllocation, annualize_emissions
        vmt = VMTAllocation.from_moves_defaults(
            source_type_id=21, calendar_year=2019, annual_vmt=12000.0,
        )
        result = annualize_emissions(rate_g_per_mile=300.0, vmt=vmt)
        expected = 300.0 * 12000.0
        assert abs(result.annual_co2_g - expected) < 1.0
        assert result.annual_co2_metric_tons == pytest.approx(
            expected / 1_000_000.0, abs=0.001
        )


# ── Technology ────────────────────────────────────────────────────────────────

class TestTechnology:
    """Test technology scenario modeling."""

    def test_constant_scenario(self):
        from neuralmoves.technology import TechnologyScenario
        ts = TechnologyScenario.constant("test", ev_fraction=0.10)
        assert ts.get_ev_fraction(2025) == 0.10
        assert ts.get_ev_fraction(2050) == 0.10

    def test_linear_ramp(self):
        from neuralmoves.technology import TechnologyScenario
        ts = TechnologyScenario.linear_ev_ramp("test", 2025, 2035, 0.05, 0.30)
        assert ts.get_ev_fraction(2025) == pytest.approx(0.05)
        assert ts.get_ev_fraction(2030) == pytest.approx(0.175)
        assert ts.get_ev_fraction(2035) == pytest.approx(0.30)

    def test_s_curve(self):
        from neuralmoves.technology import TechnologyScenario
        ts = TechnologyScenario.s_curve_ev_adoption(
            "test", 2025, 2045, 0.05, 0.80, steepness=0.5,
        )
        # S-curve should start slow, accelerate, then slow again
        assert ts.get_ev_fraction(2025) > 0.0
        assert ts.get_ev_fraction(2035) > ts.get_ev_fraction(2025)
        assert ts.get_ev_fraction(2045) > ts.get_ev_fraction(2035)
        # Endpoint should be close to target
        assert ts.get_ev_fraction(2045) == pytest.approx(0.80, abs=0.05)

    def test_interpolation(self):
        from neuralmoves.technology import TechnologyScenario
        ts = TechnologyScenario(
            name="sparse",
            ev_penetration_by_year={2020: 0.0, 2030: 1.0},
        )
        assert ts.get_ev_fraction(2025) == pytest.approx(0.5)

    def test_get_fleet(self):
        from neuralmoves.technology import TechnologyScenario
        ts = TechnologyScenario.constant("test", ev_fraction=0.15)
        fleet = ts.get_fleet(2019)
        assert fleet.ev_penetration == pytest.approx(0.15)


# ── Geography ─────────────────────────────────────────────────────────────────

class TestGeography:
    """Test location profiles."""

    def test_from_county_defaults(self):
        from neuralmoves.geography import LocationProfile
        loc = LocationProfile.from_county_defaults(26163)  # Wayne County, MI
        assert loc.location_id == "26163"
        assert len(loc.avg_temp_by_month) == 12
        assert len(loc.avg_humid_by_month) == 12
        assert loc.annual_vmt > 0

    def test_temperature_seasonal(self):
        """Michigan should have cold winters and warm summers."""
        from neuralmoves.geography import LocationProfile
        loc = LocationProfile.from_county_defaults(26163)
        jan_temp = loc.avg_temp_by_month[1]
        jul_temp = loc.avg_temp_by_month[7]
        assert jul_temp > jan_temp  # Summer warmer than winter

    def test_custom_location(self):
        from neuralmoves.geography import LocationProfile
        loc = LocationProfile.from_custom(
            "test", annual_vmt=100000, avg_temp_f=68.0, avg_humid_pct=50.0,
        )
        assert loc.annual_vmt == 100000
        assert all(t == 68.0 for t in loc.avg_temp_by_month.values())

    def test_invalid_county(self):
        from neuralmoves.geography import LocationProfile
        with pytest.raises(ValueError, match="No meteorology"):
            LocationProfile.from_county_defaults(99999)


# ── Scenarios ─────────────────────────────────────────────────────────────────

class TestScenarios:
    """Test scenario evaluation and comparison."""

    def test_ev_impact_analysis(self):
        from neuralmoves.scenarios import ev_impact_analysis
        result = ev_impact_analysis(
            ev_fractions=[0.0, 0.10, 0.20, 0.50],
            calendar_year=2019,
        )
        assert len(result) == 4
        assert result["fleet_rate_g_per_mile"].iloc[0] > 0
        # More EVs → lower rate
        assert result["fleet_rate_g_per_mile"].is_monotonic_decreasing
        # Reduction should be monotonically increasing
        assert result["reduction_pct"].is_monotonic_increasing
        # 50% EV should give ~50% reduction
        assert result.iloc[-1]["reduction_pct"] == pytest.approx(50.0, abs=2.0)

    def test_evaluate_scenario(self):
        from neuralmoves.scenarios import ScenarioDefinition, evaluate_scenario
        from neuralmoves.technology import TechnologyScenario
        from neuralmoves.temporal import VMTAllocation

        scenario = ScenarioDefinition(
            name="test",
            technology=TechnologyScenario.constant("test", 0.10),
            years=[2019],
            annual_vmt=12000.0,
        )
        result = evaluate_scenario(scenario)
        assert result.total_co2_metric_tons > 0
        assert len(result.yearly) == 1

    def test_compare_scenarios(self):
        from neuralmoves.scenarios import ScenarioDefinition, compare_scenarios
        from neuralmoves.technology import TechnologyScenario

        base = ScenarioDefinition(
            name="base",
            technology=TechnologyScenario.constant("base", 0.05),
            years=[2019],
            annual_vmt=12000.0,
        )
        policy = ScenarioDefinition(
            name="policy",
            technology=TechnologyScenario.constant("policy", 0.30),
            years=[2019],
            annual_vmt=12000.0,
        )
        df = compare_scenarios([base, policy])
        assert len(df) == 1
        assert "base_co2_MT" in df.columns
        assert "policy_co2_MT" in df.columns
        # Policy (more EVs) should have less CO2
        assert df["policy_co2_MT"].iloc[0] < df["base_co2_MT"].iloc[0]
