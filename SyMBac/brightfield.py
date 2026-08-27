"""Thin phase sample for the SyMBac brightfield path.

Two steps live here. ``scene_to_thickness_um`` turns a SyMBac scene into
micrometres, and ``apply_thin_phase_sample`` puts that thickness onto a
Chromatix vector field as a thin, isotropic, non-absorbing sample.

Both stop at the field immediately after the sample. There is no propagation,
objective, pupil, camera or detector here, so the field that comes back is not
a brightfield image.
"""

from __future__ import annotations

import numpy as np

__all__ = ["scene_to_thickness_um", "apply_thin_phase_sample"]


def _import_chromatix():
    """Import JAX and Chromatix, which this project ships only for Python 3.12."""
    # Imported here rather than at the top of the module so that importing
    # SyMBac.brightfield, and using the NumPy-only conversion below, still
    # works on the interpreters where JAX is not installed.
    try:
        import jax.numpy as jnp

        import chromatix.functional as cf
        from chromatix import VectorField
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "apply_thin_phase_sample requires JAX and Chromatix, which this "
            "project installs only for Python 3.12. Run it from the Python "
            "3.12 Pixi environment."
        ) from exc
    return cf, jnp, VectorField


def _finite_2d_array(values, name):
    """Return ``values`` as a finite 2D float array, or raise."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional array, got {array.ndim} "
            f"dimension(s) with shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_scale(value, name):
    """Return ``value`` as a positive finite float, or raise."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}."
        ) from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return scale


def scene_to_thickness_um(opl_scene, *, pix_mic_conv_um, resize_amount):
    """Convert a SyMBac scene into projected thickness in micrometres.

    ``OPL_scene`` is a legacy name. What it holds is projected geometric
    thickness measured in supersampled simulation pixels, not an optical path
    length, so this is only a change of units: one supersampled pixel is
    ``pix_mic_conv_um / resize_amount`` micrometres. Refractive-index contrast
    does not belong in this step and is not used here.

    Parameters
    ----------
    opl_scene : array_like
        2D scene of projected thickness, in supersampled simulation pixels.
        Must be finite and non-negative.
    pix_mic_conv_um : float
        Pixel size of the final, non-supersampled image, in micrometres per
        pixel. Must be positive.
    resize_amount : float
        Supersampling factor of the simulation, dimensionless. Must be
        positive.

    Returns
    -------
    numpy.ndarray
        Thickness in micrometres, same shape as ``opl_scene``. This is a new
        array, so the input is left untouched.
    """
    scene = _finite_2d_array(opl_scene, "opl_scene")
    if np.any(scene < 0.0):
        raise ValueError(
            "opl_scene must be non-negative because it is a projected thickness."
        )
    pixel_size_um = _positive_scale(pix_mic_conv_um, "pix_mic_conv_um")
    supersampling = _positive_scale(resize_amount, "resize_amount")
    return scene * pixel_size_um / supersampling


def apply_thin_phase_sample(field, thickness_um, *, refractive_index_difference):
    """Apply a thin, isotropic, pure-phase sample to a Chromatix vector field.

    The sample multiplies the field by ``exp(1j * phase_rad)``, where

    ``phase_rad = 2 * pi * refractive_index_difference * thickness_um /
    wavelength_vacuum_um``

    That factor has modulus one, so the sample is non-absorbing, and the same
    real phase reaches all three components, so intensity, power and
    polarization come out unchanged. The vacuum wavelength is taken from the
    field itself, which means it cannot disagree with the field the phase is
    applied to.

    Parameters
    ----------
    field : chromatix.VectorField
        Monochromatic Chromatix vector field, components ordered ``[z, y, x]``.
    thickness_um : array_like
        2D projected thickness in micrometres, with the same spatial shape as
        ``field``. Must be finite and non-negative.
    refractive_index_difference : float
        Refractive-index contrast between sample and surrounding medium,
        dimensionless. May be negative, and must be finite.

    Returns
    -------
    chromatix.VectorField
        A new field immediately after the sample, same spatial shape and same
        ``[z, y, x]`` component order. Neither input is modified.
    """
    cf, jnp, VectorField = _import_chromatix()

    if not isinstance(field, VectorField):
        raise TypeError(
            "field must be a monochromatic Chromatix VectorField, got "
            f"{type(field).__name__}."
        )

    thickness = _finite_2d_array(thickness_um, "thickness_um")
    if np.any(thickness < 0.0):
        raise ValueError("thickness_um must be non-negative.")

    if tuple(field.spatial_shape) != thickness.shape:
        raise ValueError(
            "thickness_um and field must have the same spatial shape, got "
            f"{thickness.shape} and {tuple(field.spatial_shape)}."
        )

    try:
        delta_n = float(refractive_index_difference)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "refractive_index_difference must be a real number, got "
            f"{type(refractive_index_difference).__name__}."
        ) from exc
    if not np.isfinite(delta_n):
        raise ValueError(
            "refractive_index_difference must be finite, got "
            f"{refractive_index_difference!r}."
        )

    # Read the wavelength from the field instead of taking it as an argument.
    wavelengths_um = np.asarray(field.spectrum.wavelength, dtype=float).reshape(-1)
    if wavelengths_um.size != 1:
        raise ValueError(
            f"field must be monochromatic, got {wavelengths_um.size} wavelengths."
        )
    wavelength_vacuum_um = float(wavelengths_um[0])
    if not np.isfinite(wavelength_vacuum_um) or wavelength_vacuum_um <= 0.0:
        raise ValueError(
            "the vacuum wavelength read from field must be positive and finite, "
            f"got {wavelength_vacuum_um!r} um."
        )

    phase_rad = 2 * np.pi * delta_n * thickness / wavelength_vacuum_um
    # The phase is already worked out for this one wavelength, so Chromatix
    # should not rescale it again: spectrally_modulate stays off.
    return cf.phase_change(field, jnp.asarray(phase_rad), spectrally_modulate=False)
