import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from functools import lru_cache
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord, EarthLocation, FK5
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clip
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks
from numpy.polynomial.legendre import legval


SPEC_DIR = '/Users/sashamintz/Desktop/Carnegie-USC/test_willman/Collated_Willman1_Spectra/'
RESULTS_DIR = '/Users/sashamintz/Desktop/Carnegie-USC/test_willman/Other/'
TEMPLATE_DIR = '/Users/sashamintz/Desktop/Carnegie-USC/test_willman/BOSZ/'

WAVE_MIN, WAVE_MAX = 4000, 6500

LINES_REST = np.array([
    4101.73, # H delta
    4226.73, # Ca I
    4340.46, # H gamma
    4383.55, # Fe I
    4861.33, # H beta
    5167.32, 5172.68, 5183.60, # Mg b triplet
    5269.54, 5328.04, # Fe I
    5528.41, # Mg I
    5889.95, 5895.92, # Na D
    6122.22, 6141.71, 6162.17, # Ca I, Ba II, Ca I
    6439.08, # Ca I
    ])


def extract_object(tucana_file):
    match = re.search(r"Wil1-(\d+)", tucana_file)
    if not match:
        return "Unknown"

    obj_str = match.group(1)

    for suffix in ["_CRCM", "_XYZ"]:
        if suffix in obj_str:
            obj_str = obj_str.split(suffix)[0]

    if obj_str.isdigit():
        return obj_str

    parts = obj_str.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return "_".join(parts[:2])

    return obj_str


def cat_delta_chi2(wave_obs, flux_obs, flux_err, model_flux, rv,
                   rv_lines=None, window=15.0):
    """
    Detection statistic: how much better does the template do than
    a flat/smooth continuum-only model in the line windows?
    Large positive = real lines detected. ~0 = fitting noise.
    """
    if rv_lines is None:
        rv_lines = rv
    c_kms = 299792.458
    cat_rest = LINES_REST
    cat_shifted = cat_rest * (1.0 + rv_lines / c_kms)

    wave_obs = np.asarray(wave_obs, dtype=float)
    flux_obs = np.asarray(flux_obs, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)

    line_mask = np.zeros_like(wave_obs, dtype=bool)
    for lam in cat_shifted:
        line_mask |= np.abs(wave_obs - lam) <= window

    valid = line_mask & (flux_err > 0) & np.isfinite(flux_obs) \
            & np.isfinite(model_flux) & np.isfinite(flux_err)
    n_pix = int(valid.sum())
    if n_pix < 5:
        return np.nan, np.nan, n_pix, np.nan, np.nan

    near = np.zeros_like(wave_obs, dtype=bool)
    for lam in cat_shifted:
        near |= np.abs(wave_obs - lam) <= 5 * window
    cont_mask = near & ~line_mask & (flux_err > 0) & np.isfinite(flux_obs)
    if cont_mask.sum() < 5:
        return np.nan, np.nan, n_pix, np.nan, np.nan
    continuum_level = np.median(flux_obs[cont_mask])

    chi2_template = np.sum(((flux_obs[valid] - model_flux[valid]) / flux_err[valid])**2)
    chi2_null = np.sum(((flux_obs[valid] - continuum_level) / flux_err[valid])**2)

    delta_chi2 = float(chi2_null - chi2_template)
    red_chi2_template = float(chi2_template / max(n_pix - 4, 1))
    return delta_chi2, red_chi2_template, n_pix, chi2_null, chi2_template


def plot_template_vs_continuum(wave_obs, flux_obs, flux_err, model_flux, rv, chi2_null, chi2_template,
                                rv_lines=None, window=15.0, beta=None,
                                object_id=None):
    if rv_lines is None:
        rv_lines = rv

    c_kms = 299792.458
    cat_rest = LINES_REST
    cat_shifted = cat_rest * (1.0 + rv_lines / c_kms)

    wave_obs = np.asarray(wave_obs, dtype=float)
    flux_obs = np.asarray(flux_obs, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)

    line_mask = np.zeros_like(wave_obs, dtype=bool)
    for lam in cat_shifted:
        line_mask |= np.abs(wave_obs - lam) <= window

    near = np.zeros_like(wave_obs, dtype=bool)
    for lam in cat_shifted:
        near |= np.abs(wave_obs - lam) <= 5 * window
    cont_mask = near & ~line_mask & (flux_err > 0) & np.isfinite(flux_obs)

    if cont_mask.sum() < 5:
        print("Not enough continuum pixels to compute null model.")
        return
    continuum_level = np.median(flux_obs[cont_mask])

    valid = line_mask & (flux_err > 0) & np.isfinite(flux_obs) \
            & np.isfinite(model_flux) & np.isfinite(flux_err)

    flux_err_plot = flux_err.copy()
    bad_err = np.isclose(flux_err_plot, 1e10, rtol=1e-3)
    flux_err_plot[bad_err] = 1e3

    plt.figure(figsize=(12, 5))

    for i, lam in enumerate(cat_shifted):
        plt.axvspan(lam - window, lam + window, color="red", alpha=0.15,
                    label="Line window" if i == 0 else None)
        plt.axvline(lam, color="red", linestyle="--", alpha=0.5, lw=0.8)

    plt.plot(wave_obs, flux_obs, color="C0", lw=0.8,
              label=f"Object {object_id}" if object_id is not None else "Observed")
    plt.fill_between(wave_obs, flux_obs - flux_err_plot, flux_obs + flux_err_plot,
                      color="C0", alpha=0.25, linewidth=0,
                      label=r"$\pm1\sigma$ error")

    plt.plot(wave_obs, model_flux, color="C1", lw=1.2, label=rf"Template × Continuum ($\chi^2$ = {chi2_template:.3f})")
    plt.axhline(continuum_level, color="black", linestyle=":", lw=1.5,
                label=rf"Continuum-only null ($\chi^2$ = {chi2_null:.3f})")

    plt.plot(wave_obs[valid], flux_obs[valid], ".", color="red", ms=3,
             label=f"Pixels used (N={valid.sum()})")

    title = "Template fit vs. continuum-only null"
    if object_id is not None:
        title = f"Object {object_id} – " + title
    if beta is not None:
        title += f", Beta (Δχ²) = {beta:.2f}"
    plt.title(title)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()


def cat_reduced_chi2(wave_obs, flux_obs, flux_err, model_flux, rv, rv_lines=None,
                     window=5.0, n_params=4):
    c_kms = 299792.458
    cat_rest = LINES_REST

    if rv_lines is None:
        rv_lines = rv
    cat_shifted = cat_rest * (1.0 + rv_lines / c_kms)

    wave_obs = np.asarray(wave_obs, dtype=float)
    flux_obs = np.asarray(flux_obs, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)

    line_mask = np.zeros_like(wave_obs, dtype=bool)
    for lam in cat_shifted:
        line_mask |= np.abs(wave_obs - lam) <= window

    valid = line_mask & (flux_err > 0) & np.isfinite(flux_obs) \
            & np.isfinite(model_flux) & np.isfinite(flux_err)

    n_pix = int(valid.sum())
    dof = n_pix - n_params
    if dof <= 0:
        return np.nan, n_pix

    resid = (flux_obs[valid] - model_flux[valid]) / flux_err[valid]
    red_chi2 = float(np.sum(resid**2) / dof)
    return red_chi2, n_pix


def plot_cat_chi2_regions(wave_obs, flux_obs, model_flux, rv,
                          rv_lines=None, window=15.0, object_id=None,
                          cat_chi2=None, flux_err=None):
    if rv_lines is None:
        rv_lines = rv

    c_kms = 299792.458
    cat_rest = LINES_REST
    cat_shifted = cat_rest * (1.0 + rv_lines / c_kms)

    wave_obs = np.asarray(wave_obs, dtype=float)
    flux_obs = np.asarray(flux_obs, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)

    plt.figure(figsize=(12, 5))

    for i, lam in enumerate(cat_shifted):
        plt.axvspan(lam - window, lam + window, color="red", alpha=0.2,
                    label="Line chi2 region" if i == 0 else None)
        plt.axvline(lam, color="red", linestyle="--", alpha=0.5, lw=0.8)

    plt.plot(wave_obs, flux_obs, color="C0", lw=0.8,
             label=f"Object {object_id}" if object_id is not None else "Observed")
    plt.plot(wave_obs, model_flux, color="C1", lw=0.8, label="Template × Continuum")

    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        line_mask = np.zeros_like(wave_obs, dtype=bool)
        for lam in cat_shifted:
            line_mask |= np.abs(wave_obs - lam) <= window
        valid = line_mask & (flux_err > 0) & np.isfinite(flux_obs) \
                & np.isfinite(model_flux) & np.isfinite(flux_err)
        plt.plot(wave_obs[valid], flux_obs[valid], ".", color="red", ms=3,
                 label=f"Pixels used (N={valid.sum()})")

    title = "Line chi2 regions"
    if object_id is not None:
        title = f"Object {object_id} – " + title
    if cat_chi2 is not None:
        title += f", Line Red Chi2 = {cat_chi2:.3f}"
    plt.title(title)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()


def add_CaT_lines(rv):
    c = 299792.458
    CaT_lines = np.array(LINES_REST)
    shifted_CaT_lines = CaT_lines * (1 + (rv / c))
    print(shifted_CaT_lines)
    return shifted_CaT_lines


def _scalar0(x):
    a = np.asarray(x, dtype=object).ravel()
    if a.size == 0:
        raise ValueError("Empty input where scalar expected.")
    v = a[0]
    if isinstance(v, (list, tuple, np.ndarray)):
        v = np.asarray(v, dtype=object).ravel()[0]
    return v


def get_heliocentric_correction(ra, dec, date):
    ra_val = float(_scalar0(ra))
    dec_val = float(_scalar0(dec))
    date_val = str(_scalar0(date))

    time = Time(date_val, format="isot", scale="utc")
    location = EarthLocation.from_geodetic(lon=-70.40*u.deg, lat=24.62*u.deg, height=2635*u.m)

    coord = SkyCoord(ra=ra_val*u.deg, dec=dec_val*u.deg, frame=FK5(equinox=Time("J2000")))

    corr = coord.radial_velocity_correction(kind="barycentric", obstime=time, location=location)
    return corr.to_value(u.km/u.s)


def estimate_object_continuum(spectrum, mask, window=50, sigma=2.0):
    spectrum = np.asarray(spectrum)
    x = np.arange(len(spectrum))
    valid = ~mask
    flux = spectrum[valid]

    clipped = sigma_clip(flux, sigma=sigma, maxiters=5)
    keep = ~clipped.mask

    rolled = minimum_filter1d(flux, size=window, mode="nearest")
    residual = flux - rolled
    broad_dip_mask = residual < -np.std(residual)

    good_mask = keep & ~broad_dip_mask
    x_good = x[valid][good_mask]
    y_good = flux[good_mask]

    interp = interp1d(x_good, y_good, kind="linear", fill_value="extrapolate")
    continuum = interp(x)
    return continuum


def shift_and_continuum(rv, b, c, d, e, f, g, h, wave_obs, wave_template, flux_template,
                        wave_ref_min, wave_ref_max):
    shifted_wave = wave_template * (1 + rv / 299792.458)
    interp_func = interp1d(shifted_wave, flux_template, kind='quadratic',
                           bounds_error=False, fill_value=1)
    shifted_flux = interp_func(wave_obs)

    span = wave_ref_max - wave_ref_min
    x = 2.0 * (wave_obs - wave_ref_min) / span - 1.0

    P0 = np.ones_like(x)
    P1 = x
    P2 = 0.5 * (3.0 * x**2 - 1.0)
    P3 = 0.5 * (5.0 * x**3 - 3.0 * x)
    P4 = (35.0 * x**4 - 30.0 * x**2 + 3.0) / 8.0
    P5 = (63.0 * x**5 - 70.0 * x**3 + 15.0 * x) / 8.0
    P6 = (231.0 * x**6 - 315.0 * x**4 + 105.0 * x**2 - 5.0) / 16.0

    continuum = (b * P0) + (c * P1) + (d * P2) + (e * P3) + (f * P4) + (g * P5) + (h * P6)
    return shifted_flux * continuum


def _find_object_fits(fits_folder: str, object_id: str):
    if not os.path.isdir(fits_folder):
        return None
    for f in sorted(os.listdir(fits_folder)):
        if f.lower().endswith(".fits") and extract_object(f) == str(object_id):
            return os.path.join(fits_folder, f)
    return None


def _load_willman_spectrum(fits_file):
    with fits.open(fits_file) as hdu:
        data = hdu[1].data
        hdr = hdu[0].header
        good = data["mask"] == 1

        wave_obs = np.asarray(data["wave"][good], dtype=float)
        flux = np.asarray(data["flux"][good], dtype=float)
        err = np.asarray(data["sigma"][good], dtype=float)
        ra = hdr["RA_OBJ"]
        dec = hdr["DEC_OBJ"]
        date = hdr["DATE-OBS"]

    return wave_obs, flux, err, ra, dec, date


def _find_template_csvs(galaxy: str):
    pattern = re.compile(rf"{re.escape(galaxy)}_(\d+)_(\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.csv")
    out = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        m = pattern.fullmatch(f)
        if m:
            label = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
            out.append((label, os.path.join(RESULTS_DIR, f),
                        int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def _load_runs(galaxy: str):
    templates = _find_template_csvs(galaxy)
    if not templates:
        raise FileNotFoundError(f"No {galaxy}_<Teff>_<logg>_<FeH>.csv files found in {RESULTS_DIR}")

    print("\nTemplate runs found:")
    for label, path, *_ in templates:
        print(f"  {label}: {os.path.basename(path)}")

    runs = {}
    for label, path, teff, logg, feh in templates:
        df_t = pd.read_csv(path, dtype={"Objects": str}).drop_duplicates("Objects").set_index("Objects")
        runs[label] = (df_t, teff, logg, feh)
    return runs


@lru_cache(maxsize=None)
def _load_template(teff, logg, feh):
    matches = sorted(glob.glob(os.path.join(TEMPLATE_DIR, f"t{teff}_g{logg:+.1f}_m{feh:+.2f}_*.npz")))
    if not matches:
        raise FileNotFoundError(f"No template .npz for Teff={teff}, logg={logg}, [Fe/H]={feh} in {TEMPLATE_DIR}")
    with np.load(matches[0]) as tpl:
        return tpl["wave"], tpl["flux"]


def _apply_mcmc_cuts(wave_obs, flux, err, wave_template, flux_template):
    cut_mask = (wave_obs >= WAVE_MIN) & (wave_obs <= WAVE_MAX)
    wave_obs = wave_obs[cut_mask]
    flux = flux[cut_mask]
    err = err[cut_mask]

    min_wave = max(np.min(wave_template), np.min(wave_obs))
    max_wave = min(np.max(wave_template), np.max(wave_obs))

    mask_template = (wave_template >= min_wave) & (wave_template <= max_wave)
    mask_obs = (wave_obs >= min_wave) & (wave_obs <= max_wave)

    wave_template = wave_template[mask_template]
    flux_template = flux_template[mask_template]
    wave_obs = wave_obs[mask_obs]
    flux = flux[mask_obs]
    err = err[mask_obs]

    wave_ref_min = float(np.nanmin(wave_obs))
    wave_ref_max = float(np.nanmax(wave_obs))
    if not np.isfinite(wave_ref_max - wave_ref_min) or wave_ref_max <= wave_ref_min:
        wave_ref_max = wave_ref_min + 1.0

    return wave_obs, flux, err, wave_template, flux_template, wave_ref_min, wave_ref_max


def _model_for(row, wave_raw, flux_raw, err_raw, teff, logg, feh):
    wave_t, flux_t = _load_template(teff, logg, feh)
    (wave_obs, flux, err,
     wave_template, flux_template,
     wave_ref_min, wave_ref_max) = _apply_mcmc_cuts(wave_raw, flux_raw, err_raw, wave_t, flux_t)

    model_flux = shift_and_continuum(float(row["RV"]), float(row["b val"]),
                                     float(row["c val"]), float(row["d val"]),
                                     float(row.get("e val", 0.0)), float(row.get("f val", 0.0)),
                                     float(row.get("g val", 0.0)), float(row.get("h val", 0.0)),
                                     wave_obs, wave_template, flux_template,
                                     wave_ref_min, wave_ref_max)
    return wave_obs, flux, err, model_flux


def process_galaxy(galaxy: str, runs: dict) -> pd.DataFrame:
    all_objects = sorted(set().union(*[set(r[0].index) for r in runs.values()]))

    records = []
    for object_id in all_objects:
        chi2s = {}
        for label, (df_t, teff, logg, feh) in runs.items():
            if object_id in df_t.index:
                chi2s[label] = float(df_t.loc[object_id, "Reduced Chi Squared"])

        finite = {k: v for k, v in chi2s.items() if np.isfinite(v)}
        if not finite:
            print(f"[{galaxy}] No valid reduced chi2 for {object_id}, skipping.")
            continue

        best = min(finite, key=lambda k: abs(finite[k] - 1.0))
        df_t, teff, logg, feh = runs[best]
        row = df_t.loc[object_id]
        rv = float(row["RV"])

        snr = np.nan
        delta_chi2 = np.nan
        fits_file = _find_object_fits(SPEC_DIR, object_id)
        if fits_file is None:
            print(f"[{galaxy}] No spectrum file for {object_id}, S/N and Beta left blank.")
        else:
            try:
                wave_raw, flux_raw, err_raw, ra, dec, date = _load_willman_spectrum(fits_file)
                wave_obs, flux, err, model_flux = _model_for(row, wave_raw, flux_raw, err_raw,
                                                             teff, logg, feh)

                valid = (err > 0) & np.isfinite(flux) & np.isfinite(err)
                snr = np.median(np.abs(flux[valid] / err[valid])) if valid.sum() >= 10 else np.nan

                delta_chi2, red_chi2_cat, n_pix, chi2_null, chi2_template = cat_delta_chi2(wave_obs, flux, err,
                                                                                           model_flux, rv,
                                                                                           rv_lines=rv,
                                                                                           window=1.0)
            except Exception as e:
                print(f"[{galaxy}] S/N and Beta failed for {object_id} with template {best}: {e}")

        rec = {
            "Objects": object_id,
            "RV": rv,
            "RV with all HECO": float(row["RV with all HECO"]),
            "RV uncertainty": float(row["RV uncertainty"]),
            "S/N": snr,
            "Beta": delta_chi2,
            "Reduced Chi Squared": finite[best],
            "Best Template": best,
            "Best Teff": teff,
            "Best logg": logg,
            "Best [Fe/H]": feh,
            "b val": float(row["b val"]),
            "c val": float(row["c val"]),
            "d val": float(row["d val"]),
            "e val": float(row.get("e val", 0.0)),
            "f val": float(row.get("f val", 0.0)),
            "g val": float(row.get("g val", 0.0)),
            "h val": float(row.get("h val", 0.0)),
        }
        for label in runs:
            rec[f"Red Chi2 {label}"] = chi2s.get(label, np.nan)
        records.append(rec)

    df_export = pd.DataFrame(records)
    if df_export.empty:
        raise RuntimeError(f"No objects could be matched across the template CSVs. "
                           f"Check the [{galaxy}] messages printed above.")

    save_path = f"{RESULTS_DIR}Final_RV_Summary_{galaxy}.csv"
    df_export.to_csv(save_path, index=False)

    print(f"\n[{galaxy}] Saved best-template RV summary to:\n{save_path}\n")
    return df_export


def _plot_fit(galaxy, object_id, spec, row, teff, logg, feh, label, red_chi2, is_best):
    wave_raw, flux_raw, err_raw = spec
    try:
        wave_obs, flux_obs, flux_err, model_flux = _model_for(row, wave_raw, flux_raw, err_raw,
                                                              teff, logg, feh)
    except Exception as e:
        print(f"Could not build model for {object_id} with template {label}: {e}")
        return

    rv = float(row["RV"])
    rv_w_heco = float(row["RV with all HECO"])
    rv_uncert = float(row["RV uncertainty"])

    print(f"\nTemplate: {label} (Teff={teff}, logg={logg}, [Fe/H]={feh})"
          f"{'  [best fit]' if is_best else ''}")
    print(f"RV (before adding HECO): {rv:.2f} km/s")

    rv_corr = np.abs(rv - rv_w_heco)
    rv_test = rv + rv_corr

    print(f"RV corr: {rv_corr}")
    print(f"RV test: {rv_test}")

    rv_lines = rv
    cat_chi2, n_pix = cat_reduced_chi2(wave_obs, flux_obs, flux_err, model_flux, rv)
    print(f"Line-only reduced chi2: {cat_chi2:.3f} over {n_pix} pixels")

    print("Mean difference (template - obs):", np.mean(model_flux - flux_obs))
    print("Std difference (template - obs):", np.std(model_flux - flux_obs))
    print("Wavelength range:", wave_obs[0], wave_obs[-1])

    shifted_CaT_lines = add_CaT_lines(rv_lines)
    plt.figure(figsize=(10, 5))
    for i in range(len(shifted_CaT_lines)):
        if i == 0:
            plt.axvline(shifted_CaT_lines[i], color="grey", linestyle="--", label="Line positions")
        else:
            plt.axvline(shifted_CaT_lines[i], color="grey", linestyle="--")

    wave_obs = wave_obs[200:-200]
    flux_obs = flux_obs[200:-200]
    model_flux = model_flux[200:-200]

    plt.plot(wave_obs, flux_obs, label=f"Object {object_id} (Willman 1)")
    plt.plot(
        wave_obs,
        model_flux,
        label=f"BOSZ {teff} K, logg {logg}, [Fe/H] {feh} × Continuum (rv = {rv_w_heco:.2f} ± {rv_uncert:.2f} km/s)",
    )
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    tag = "Best Template" if is_best else "Template"
    plt.title(f"Object {object_id} – {galaxy}, {tag} {label}, Red Chi2 = {red_chi2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_plot_loop(galaxy: str, df_export: pd.DataFrame, runs: dict):
    print(f"\n=== Plotting mode: Galaxy={galaxy} ===")
    print("\nAvailable Objects with corresponding RV and S/N:")
    print(df_export[["Objects", "RV", "RV with all HECO", "RV uncertainty", "S/N", "Beta",
                     "Reduced Chi Squared", "Best Template"]]
          .to_string(index=False, float_format="{:.2f}".format))

    df = df_export

    while True:
        mode = input("\nWould you like to view by Object (O), or Exit (E)? ").strip().upper()
        if mode == "E":
            print(f"Done with {galaxy}.")
            return

        elif mode == "O":
            object_id = input("Enter object ID: ").strip()
            row = df[df["Objects"].astype(str) == object_id]
        else:
            print("Invalid mode. Try again.")
            continue

        if row.empty:
            print("No match found. Try again.")
            continue

        row = row.iloc[0]
        best_label = row["Best Template"]

        fits_file = _find_object_fits(SPEC_DIR, str(object_id))
        if fits_file is None:
            print(f"No Willman 1 file found for object {object_id}. Skipping...")
            continue

        wave_raw, flux_raw, err_raw, ra, dec, date = _load_willman_spectrum(fits_file)
        spec = (wave_raw, flux_raw, err_raw)

        options = []
        for label, (df_t, teff, logg, feh) in runs.items():
            if object_id in df_t.index:
                options.append((label, teff, logg, feh, float(df_t.loc[object_id, "Reduced Chi Squared"])))
        options.sort(key=lambda o: (o[1], o[2], o[3]))

        while True:
            choice = input("\nPlot with best-fit template (B) or choose a template by temperature (T)? ").strip().upper()
            if choice in ("B", "T"):
                break
            print("Please enter B or T.")

        if choice == "B":
            _plot_fit(galaxy, object_id, spec, row,
                      int(row["Best Teff"]), float(row["Best logg"]), float(row["Best [Fe/H]"]),
                      best_label, float(row["Reduced Chi Squared"]), is_best=True)
            continue

        while True:
            print(f"\nAvailable templates for object {object_id}:")
            for label, teff, logg, feh, chi2 in options:
                tag = "  <- best fit" if label == best_label else ""
                print(f"  {teff} K, logg {logg}, [Fe/H] {feh}   (Red Chi2 = {chi2:.3f}){tag}")

            sel = input("\nEnter a temperature (e.g. 4750), or 'S' to go back to stars: ").strip()
            if sel.upper() == "S":
                break

            try:
                t_req = int(float(sel))
            except ValueError:
                print("Please enter a temperature or 'S'.")
                continue

            matches = [o for o in options if o[1] == t_req]
            if not matches:
                print(f"No template at {t_req} K for this object.")
                continue

            if len(matches) > 1:
                print("\nMore than one template at that temperature:")
                for i, (label, teff, logg, feh, chi2) in enumerate(matches, 1):
                    print(f"  {i}. {teff} K, logg {logg}, [Fe/H] {feh}   (Red Chi2 = {chi2:.3f})")
                pick = input("Pick a number: ").strip()
                if not pick.isdigit() or not (1 <= int(pick) <= len(matches)):
                    print("Invalid choice.")
                    continue
                chosen = matches[int(pick) - 1]
            else:
                chosen = matches[0]

            label, teff, logg, feh, chi2 = chosen
            df_t = runs[label][0]
            _plot_fit(galaxy, object_id, spec, df_t.loc[object_id],
                      teff, logg, feh, label, chi2, is_best=(label == best_label))


galaxy = input("Enter the galaxy name used in the MCMC run (e.g. Willman1): ").strip()

print(f"\nYou selected: Galaxy={galaxy}")
runs = _load_runs(galaxy)
df_export = process_galaxy(galaxy, runs)
run_plot_loop(galaxy, df_export, runs)