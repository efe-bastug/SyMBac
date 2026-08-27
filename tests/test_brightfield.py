"""Tests for the thin phase sample core in ``SyMBac.brightfield``.

The whole file skips if JAX and Chromatix are missing, since the project only
installs them for Python 3.12. They are there in the 3.12 Pixi environment, so
these tests run rather than skip.
"""

import numpy as np
import pytest

jnp = pytest.importorskip(
    "jax.numpy",
    reason="JAX is installed only for Python 3.12 in this project.",
)
cf = pytest.importorskip(
    "chromatix.functional",
    reason="Chromatix is installed only for Python 3.12 in this project.",
)

from chromatix import VectorField  # noqa: E402  (imported after the skip guard)

from SyMBac.brightfield import (  # noqa: E402  (imported after the skip guard)
    apply_thin_phase_sample,
    scene_to_thickness_um,
)
from SyMBac.drawing import draw_scene_from_segments  # noqa: E402

WAVELENGTH_UM = 0.532
DX_UM = 0.25
PIX_MIC_CONV_UM = 0.065
RESIZE_AMOUNT = 3
DELTA_N = 0.05


def _vector_plane_wave(shape=(8, 8), amplitude=None, wavelength_um=WAVELENGTH_UM):
    """Small monochromatic vector plane wave, components ``[z, y, x]``."""
    if amplitude is None:
        amplitude = cf.linear(0.0)  # [E_z, E_y, E_x] = [0, 0, 1]
    return cf.plane_wave(
        shape, DX_UM, wavelength_um, amplitude=amplitude, scalar=False
    )


def _expected_phase_rad(thickness_um, refractive_index_difference, wavelength_um=WAVELENGTH_UM):
    """Expected phase, worked out here so the test does not reuse the code."""
    return (
        2
        * np.pi
        * refractive_index_difference
        * np.asarray(thickness_um, dtype=float)
        / wavelength_um
    )


# Task point 1: one supersampled pixel is pix_mic_conv_um / resize_amount
# micrometres, and neither axis gets swapped.
def test_scene_converts_per_supersampled_pixel_without_transposing_axes():
    scene = np.arange(6, dtype=float).reshape(2, 3)

    thickness_um = scene_to_thickness_um(
        scene, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
    )

    np.testing.assert_array_equal(
        thickness_um, scene * PIX_MIC_CONV_UM / RESIZE_AMOUNT
    )
    assert thickness_um.shape == scene.shape == (2, 3)

    micrometres_per_pixel = PIX_MIC_CONV_UM / RESIZE_AMOUNT
    assert thickness_um[1, 2] == pytest.approx(scene[1, 2] * micrometres_per_pixel)

    marker = np.zeros((2, 3))
    marker[0, 2] = 7.0
    converted = scene_to_thickness_um(
        marker, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
    )
    assert converted[0, 2] == pytest.approx(7.0 * micrometres_per_pixel)
    assert converted.sum() == pytest.approx(converted[0, 2])

    transposed = scene_to_thickness_um(
        scene.T, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
    )
    np.testing.assert_array_equal(transposed, thickness_um.T)


# Task point 2: a scene from the current draw_scene_from_segments goes
# through both functions.
def test_scene_from_draw_scene_from_segments_passes_through_both_functions():
    cells_segment_data = [
        {
            "positions": np.array([[12.0, 10.0], [18.0, 10.0]]),
            "radii": np.array([4.0, 4.0]),
            "mask_label": 1,
            "cell_id": 1,
        }
    ]

    scene, _mask = draw_scene_from_segments(cells_segment_data, (20, 30), 0, True)
    assert scene.shape == (20, 30)
    assert np.any(scene > 0.0)

    thickness_um = scene_to_thickness_um(
        scene, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
    )
    field = _vector_plane_wave(shape=scene.shape)

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=DELTA_N
    )

    assert isinstance(sampled, VectorField)
    assert sampled.u.shape == (20, 30, 3)
    assert bool(jnp.all(jnp.isfinite(sampled.u)))


# Task point 3: check against an input * exp(i * phase) worked out here in
# the test, not with Chromatix.
def test_field_matches_independent_exp_i_phase_calculation():
    shape = (4, 6)
    thickness_um = np.linspace(0.0, 1.2, num=24).reshape(shape)
    field = _vector_plane_wave(shape)

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=DELTA_N
    )

    phase_rad = _expected_phase_rad(thickness_um, DELTA_N)
    expected_u = np.asarray(field.u) * np.exp(1j * phase_rad)[..., np.newaxis]
    np.testing.assert_allclose(
        np.asarray(sampled.u), expected_u, rtol=1e-6, atol=1e-7
    )


# Task point 4: all three components get the same phase, so the
# polarization does not move.
def test_every_component_receives_the_same_phase_so_polarisation_is_unchanged():
    shape = (4, 4)
    amplitude = cf.linear(np.pi / 4)  # [0, E_y, E_x] with both transverse parts non-zero
    assert float(jnp.abs(amplitude[1])) > 0.1
    assert float(jnp.abs(amplitude[2])) > 0.1

    field = _vector_plane_wave(shape, amplitude=amplitude)
    thickness_um = np.full(shape, 0.8)

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=DELTA_N
    )

    ratio_y = np.asarray(sampled.u[..., 1]) / np.asarray(field.u[..., 1])
    ratio_x = np.asarray(sampled.u[..., 2]) / np.asarray(field.u[..., 2])
    np.testing.assert_allclose(ratio_y, ratio_x, rtol=1e-6)
    np.testing.assert_allclose(
        np.angle(ratio_x),
        _expected_phase_rad(thickness_um, DELTA_N),
        rtol=1e-5,
        atol=1e-6,
    )

    polarisation_before = np.asarray(field.u[..., 1]) / np.asarray(field.u[..., 2])
    polarisation_after = np.asarray(sampled.u[..., 1]) / np.asarray(sampled.u[..., 2])
    np.testing.assert_allclose(polarisation_after, polarisation_before, rtol=1e-6)


# Task point 5: no thickness or no index contrast means no change.
def test_zero_thickness_or_zero_index_difference_leaves_field_unchanged():
    shape = (4, 4)
    field = _vector_plane_wave(shape, amplitude=cf.linear(np.pi / 4))

    zero_thickness = apply_thin_phase_sample(
        field, np.zeros(shape), refractive_index_difference=DELTA_N
    )
    np.testing.assert_allclose(
        np.asarray(zero_thickness.u), np.asarray(field.u), rtol=0.0, atol=0.0
    )

    zero_contrast = apply_thin_phase_sample(
        field, np.full(shape, 0.8), refractive_index_difference=0.0
    )
    np.testing.assert_allclose(
        np.asarray(zero_contrast.u), np.asarray(field.u), rtol=0.0, atol=0.0
    )


# Task point 6: a pure phase sample keeps intensity and power.
def test_pure_phase_sample_preserves_intensity_and_power():
    shape = (8, 8)
    field = _vector_plane_wave(shape, amplitude=cf.linear(np.pi / 3))
    thickness_um = np.linspace(0.0, 1.5, num=64).reshape(shape)

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=DELTA_N
    )

    np.testing.assert_allclose(
        np.asarray(sampled.intensity), np.asarray(field.intensity), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(sampled.power), np.asarray(field.power), rtol=1e-6, atol=1e-9
    )


# Task point 7: type, shape, [z, y, x] components, finite values, and the
# inputs left untouched.
def test_output_type_shape_components_and_input_immutability():
    shape = (5, 7)
    field = _vector_plane_wave(shape)
    thickness_um = np.full(shape, 0.8)

    thickness_before = thickness_um.copy()
    u_before = np.asarray(field.u).copy()

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=DELTA_N
    )

    assert isinstance(sampled, VectorField)
    assert sampled.u.shape == (5, 7, 3)
    assert tuple(sampled.spatial_shape) == shape
    assert bool(jnp.all(jnp.isfinite(sampled.u)))

    # the input is x-polarised, so in [z, y, x] order only the last component
    # carries amplitude and the leading z component stays exactly zero.
    assert float(jnp.max(jnp.abs(sampled.u[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(sampled.u[..., 1]))) == 0.0
    assert float(jnp.max(jnp.abs(sampled.u[..., 2]))) > 0.0

    np.testing.assert_array_equal(thickness_um, thickness_before)
    np.testing.assert_array_equal(np.asarray(field.u), u_before)


# Extra: the wavelength really does come from the field, not an argument.
def test_phase_uses_the_wavelength_stored_in_the_field():
    shape = (4, 4)
    thickness_um = np.full(shape, 0.8)

    for wavelength_um in (0.405, 0.532, 0.660):
        field = _vector_plane_wave(shape, wavelength_um=wavelength_um)
        sampled = apply_thin_phase_sample(
            field, thickness_um, refractive_index_difference=DELTA_N
        )
        ratio = np.asarray(sampled.u[..., 2]) / np.asarray(field.u[..., 2])
        np.testing.assert_allclose(
            np.angle(ratio),
            _expected_phase_rad(thickness_um, DELTA_N, wavelength_um=wavelength_um),
            rtol=1e-5,
            atol=1e-6,
        )


def test_refractive_index_difference_may_be_negative():
    shape = (4, 4)
    field = _vector_plane_wave(shape)
    thickness_um = np.full(shape, 0.8)

    sampled = apply_thin_phase_sample(
        field, thickness_um, refractive_index_difference=-DELTA_N
    )

    ratio = np.asarray(sampled.u[..., 2]) / np.asarray(field.u[..., 2])
    np.testing.assert_allclose(
        np.angle(ratio),
        _expected_phase_rad(thickness_um, -DELTA_N),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sampled.intensity), np.asarray(field.intensity), rtol=1e-6, atol=1e-9
    )


# Task point 8: bad input has to fail with a clear message.
@pytest.mark.parametrize(
    "scene, message",
    [
        (np.zeros((2, 2, 2)), "two-dimensional"),
        (np.zeros(4), "two-dimensional"),
        (np.array([[0.0, np.nan]]), "finite"),
        (np.array([[0.0, np.inf]]), "finite"),
        (np.array([[0.0, -1.0]]), "non-negative"),
    ],
)
def test_invalid_scene_raises(scene, message):
    with pytest.raises(ValueError, match=message):
        scene_to_thickness_um(
            scene, pix_mic_conv_um=PIX_MIC_CONV_UM, resize_amount=RESIZE_AMOUNT
        )


@pytest.mark.parametrize(
    "pix_mic_conv_um, resize_amount, message",
    [
        (0.0, RESIZE_AMOUNT, "pix_mic_conv_um"),
        (-0.065, RESIZE_AMOUNT, "pix_mic_conv_um"),
        (np.nan, RESIZE_AMOUNT, "pix_mic_conv_um"),
        (PIX_MIC_CONV_UM, 0, "resize_amount"),
        (PIX_MIC_CONV_UM, -3, "resize_amount"),
        (PIX_MIC_CONV_UM, np.inf, "resize_amount"),
    ],
)
def test_invalid_scale_values_raise(pix_mic_conv_um, resize_amount, message):
    with pytest.raises(ValueError, match=message):
        scene_to_thickness_um(
            np.ones((2, 2)),
            pix_mic_conv_um=pix_mic_conv_um,
            resize_amount=resize_amount,
        )


@pytest.mark.parametrize(
    "thickness_um, message",
    [
        (np.zeros((2, 2, 2)), "two-dimensional"),
        (np.array([[0.0, np.nan], [0.0, 0.0]]), "finite"),
        (np.array([[0.0, -0.5], [0.0, 0.0]]), "non-negative"),
    ],
)
def test_invalid_thickness_map_raises(thickness_um, message):
    field = _vector_plane_wave((2, 2))
    with pytest.raises(ValueError, match=message):
        apply_thin_phase_sample(
            field, thickness_um, refractive_index_difference=DELTA_N
        )


@pytest.mark.parametrize("refractive_index_difference", [np.nan, np.inf, -np.inf])
def test_non_finite_refractive_index_difference_raises(refractive_index_difference):
    field = _vector_plane_wave((4, 4))
    with pytest.raises(ValueError, match="refractive_index_difference must be finite"):
        apply_thin_phase_sample(
            field,
            np.zeros((4, 4)),
            refractive_index_difference=refractive_index_difference,
        )


def test_field_and_thickness_shape_mismatch_raises():
    field = _vector_plane_wave((4, 4))
    with pytest.raises(ValueError, match="same spatial shape"):
        apply_thin_phase_sample(
            field, np.zeros((4, 5)), refractive_index_difference=DELTA_N
        )


def test_scalar_field_is_rejected():
    scalar_field = cf.plane_wave((4, 4), DX_UM, WAVELENGTH_UM)
    with pytest.raises(TypeError, match="VectorField"):
        apply_thin_phase_sample(
            scalar_field, np.zeros((4, 4)), refractive_index_difference=DELTA_N
        )
