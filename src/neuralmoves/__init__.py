# SPDX-FileCopyrightText: 2024-present Edgar Ramirez Sanchez <edgarrs@mit.edu>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__
from .emission_calculation import get_emissions

__all__ = ["get_emissions", "__version__"]
