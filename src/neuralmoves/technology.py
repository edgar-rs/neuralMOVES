"""Layer 4: Technology scenarios — model EV adoption and fleet transitions.

Provides tools for defining technology adoption curves (linear, S-curve)
and generating year-by-year fleet compositions.

Example
-------
>>> from neuralmoves.technology import TechnologyScenario
>>> baseline = TechnologyScenario.linear_ev_ramp("base", 2025, 2035, 0.05, 0.05)
>>> policy = TechnologyScenario.linear_ev_ramp("policy", 2025, 2035, 0.05, 0.30)
>>> print(f"2030 baseline EV: {baseline.get_ev_fraction(2030):.1%}")
>>> print(f"2030 policy EV:   {policy.get_ev_fraction(2030):.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TechnologyScenario:
    """A technology adoption scenario over time.

    Attributes
    ----------
    name : str
        Descriptive name for this scenario.
    ev_penetration_by_year : dict[int, float]
        EV VMT share (0-1) for each year. Years between defined points
        are interpolated when accessed via get_ev_fraction().
    """

    name: str
    ev_penetration_by_year: dict[int, float] = field(default_factory=dict)

    # ── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def constant(
        cls,
        name: str,
        ev_fraction: float,
        years: Optional[list[int]] = None,
    ) -> "TechnologyScenario":
        """Create a scenario with constant EV penetration."""
        if years is None:
            years = list(range(2020, 2061))
        return cls(
            name=name,
            ev_penetration_by_year={y: ev_fraction for y in years},
        )

    @classmethod
    def linear_ev_ramp(
        cls,
        name: str,
        start_year: int,
        end_year: int,
        start_pct: float,
        end_pct: float,
    ) -> "TechnologyScenario":
        """Create a linear EV adoption ramp.

        Parameters
        ----------
        name : str
            Scenario name.
        start_year, end_year : int
            First and last year of the ramp.
        start_pct, end_pct : float
            EV fraction (0-1) at start and end.
        """
        years = range(start_year, end_year + 1)
        n = end_year - start_year
        if n == 0:
            fracs = {start_year: start_pct}
        else:
            fracs = {
                y: start_pct + (end_pct - start_pct) * (y - start_year) / n
                for y in years
            }
        return cls(name=name, ev_penetration_by_year=fracs)

    @classmethod
    def s_curve_ev_adoption(
        cls,
        name: str,
        start_year: int,
        end_year: int,
        start_pct: float,
        end_pct: float,
        steepness: float = 0.5,
    ) -> "TechnologyScenario":
        """Create an S-curve (logistic) EV adoption trajectory.

        Parameters
        ----------
        name : str
            Scenario name.
        start_year, end_year : int
            Year range.
        start_pct, end_pct : float
            EV fraction (0-1) at endpoints.
        steepness : float
            Controls how steep the S-curve is (0.1 = gradual, 1.0 = sharp).
        """
        years = range(start_year, end_year + 1)
        mid_year = (start_year + end_year) / 2.0
        span = end_year - start_year
        if span == 0:
            return cls(name=name, ev_penetration_by_year={start_year: start_pct})

        k = steepness * 12.0 / span  # scale steepness to year range

        fracs = {}
        for y in years:
            # Logistic function normalized to [0, 1]
            t = k * (y - mid_year)
            sigmoid = 1.0 / (1.0 + np.exp(-t))
            frac = start_pct + (end_pct - start_pct) * sigmoid
            fracs[y] = float(np.clip(frac, 0.0, 1.0))
        return cls(name=name, ev_penetration_by_year=fracs)

    # ── Access ───────────────────────────────────────────────────────────

    def get_ev_fraction(self, year: int) -> float:
        """Get EV penetration for a given year (interpolates between defined years)."""
        if year in self.ev_penetration_by_year:
            return self.ev_penetration_by_year[year]

        defined_years = sorted(self.ev_penetration_by_year.keys())
        if not defined_years:
            return 0.0

        # Clamp to range
        if year <= defined_years[0]:
            return self.ev_penetration_by_year[defined_years[0]]
        if year >= defined_years[-1]:
            return self.ev_penetration_by_year[defined_years[-1]]

        # Linear interpolation between nearest defined years
        for i in range(len(defined_years) - 1):
            y0, y1 = defined_years[i], defined_years[i + 1]
            if y0 <= year <= y1:
                f0 = self.ev_penetration_by_year[y0]
                f1 = self.ev_penetration_by_year[y1]
                t = (year - y0) / (y1 - y0)
                return f0 + (f1 - f0) * t

        return 0.0

    def get_fleet(
        self,
        year: int,
        base_fleet: Optional["FleetComposition"] = None,
    ) -> "FleetComposition":
        """Get fleet composition for a given year.

        Parameters
        ----------
        year : int
            Calendar year.
        base_fleet : FleetComposition or None
            If provided, adjusts this fleet's EV penetration.
            If None, builds from MOVES defaults for the given year.
        """
        from .fleet import FleetComposition

        ev_frac = self.get_ev_fraction(year)

        if base_fleet is not None:
            return base_fleet.with_ev_penetration(ev_frac)

        return FleetComposition.from_moves_defaults(
            calendar_year=year,
            ev_penetration=ev_frac,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the adoption curve as a DataFrame."""
        rows = sorted(self.ev_penetration_by_year.items())
        return pd.DataFrame(rows, columns=["year", "ev_fraction"])

    @property
    def years(self) -> list[int]:
        return sorted(self.ev_penetration_by_year.keys())

    def __repr__(self) -> str:
        yrs = self.years
        if len(yrs) >= 2:
            return (
                f"TechnologyScenario({self.name!r}, "
                f"{yrs[0]}-{yrs[-1]}, "
                f"ev={self.get_ev_fraction(yrs[0]):.1%}→"
                f"{self.get_ev_fraction(yrs[-1]):.1%})"
            )
        return f"TechnologyScenario({self.name!r})"


__all__ = ["TechnologyScenario"]
