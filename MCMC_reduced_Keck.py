#!/usr/bin/env python3
import os
import re
import time
import numpy as np
import pandas as pd
# Make plotting safe inside subprocesses (no GUI)
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.special import gammaln
import emcee
import corner
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from numpy.polynomial.legendre import legval, legfit
import traceback


start_time = time.time()



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
    # print('new version test')
    return shifted_flux * continuum


def log_likelihood(rv, b, c, d, e, f, g, h, wave_ref_median, wave_ref_std, wave_template, flux_template, wave_obs, flux_obs, flux_err):
    """
    Poisson log-likelihood.
    """
    model_flux = np.maximum(
        shift_and_continuum(rv, b, c, d, e, f, g, h, wave_obs, wave_template, flux_template, wave_ref_median, wave_ref_std),
        1e-10
    )

    model_flux = model_flux[200:-200]
    wave_obs = wave_obs[200:-200]
    wave_template = wave_template[200:-200]
    flux_obs = flux_obs[200:-200]
    flux_err = flux_err[200:-200]

    residuals = flux_obs - model_flux
    ll = -0.5 * np.sum((residuals / flux_err)**2 + np.log(2 * np.pi * flux_err**2))
    return ll


def log_prior(rv, b, c, d, e, f, g, h, b_min, b_max, c_min, c_max, d_min, d_max,
              e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max):
    """
    Top-hat priors.
    """
    if (-100 < rv < 100) and (b_min < b < b_max) and (c_min < c < c_max) and (d_min < d < d_max) \
            and (e_min < e < e_max) and (f_min < f < f_max) and (g_min < g < g_max) and (h_min < h < h_max):
        return 0.0
    return -np.inf


def log_probability(params, wave_template, flux_template, wave_obs, flux_obs, b_min, b_max, c_min, c_max, d_min, d_max,
                    e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max, wave_ref_median, wave_ref_std, flux_err):
    rv, b, c, d, e, f, g, h = params
    lp = log_prior(rv, b, c, d, e, f, g, h, b_min, b_max, c_min, c_max, d_min, d_max,
                   e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(rv, b, c, d, e, f, g, h, wave_ref_median, wave_ref_std, wave_template, flux_template, wave_obs, flux_obs, flux_err)


def extract_fiber_number(tucana_file):
    match = re.search(r"Wil1-(\d+)", tucana_file)
    return match.group(1) if match else "Unknown"


def extract_priority(tucana_file):
    match = re.search(r"PRI=(\d+)", tucana_file)
    return match.group(1) if match else "Unknown"


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


def get_heliocentric_correction(ra, dec, date):
    """
    Heliocentric correction in km/s.
    """
    ra = float(ra)
    dec = float(dec)
    date = str(date)
    print(date)
    # exptime = float(exptime[0])

    t = Time(date, format='isot', scale='utc')

    # shift to mid-exposure time for a potentially more accurate correction ?
    # t_mid = t_start + (exptime / 2) * u.s

    location = EarthLocation.from_geodetic(lon=-70.40*u.deg, lat=24.62*u.deg, height=2635*u.m)
    object_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    corr = object_coord.radial_velocity_correction(kind='barycentric', obstime=t, location=location)
    return corr.to(u.km/u.s).value.item()

# ============================================================================================== #
# def get_data(tucana_file, galaxy, observation_date, correction):
#     """
#     Load observed file + reference template star (HD12256), trim arrays, and return needed payload.
#     """
#     # File for reference star HD12256
#     # hd_file = '/Users/sashamintz/Desktop/Carnegie-USC/data/My Reduced Data/TucIII/08-07-2021/TucIII-ep1_SCIENCE_RBNSPEC_IDP_PRI=9_OBJ=TucIII_1028778334.fits'
#     hd_file = '/Users/sashamintz/Desktop/Carnegie-USC/data/My Reduced Data/HD122563/04-29-2016/HD122563-ep1_SCIENCE_RBNSPEC_IDP_PRI=9_OBJ=HD_122563.fits'
#     if galaxy == "LeoIV":
#         hd_file = "/Users/sashamintz/Desktop/Carnegie-USC/data/My Reduced Data/LeoIV/05-09-2010A/LeoIV-ep1_SCIENCE_RBNSPEC_IDP_PRI=9_OBJ=Leo4_1146.fits"

#     fiber_number = extract_fiber_number(tucana_file)
#     priority = extract_priority(tucana_file)
#     object_id = extract_object(tucana_file)

#     # print('opening fits file')
#     try:
#         with fits.open(tucana_file) as hdu:
#             asn_table = Table(hdu[1].data)
#             wave_tucana = np.array(asn_table['WAVE'])
#             flux_sky_sub = np.array(asn_table['FLUX_REDUCED'])
#             flux_err = np.array(asn_table['ERR_REDUCED'])
#             rv_wave_uncert = np.array(asn_table['RV_ERR'][0])
#             # print(rv_wave_uncert)
#             # print(np.shape(wave_tucana), np.shape(flux_sky_sub), np.shape(flux_err))

#             ra = asn_table['RA'].tolist()
#             dec = asn_table['DEC'].tolist()
#             date = asn_table['DATE-OBS'].tolist()
#             # exptime = asn_table['EXPTIME'].tolist()
#         # print('opening hd file')
#         with fits.open(hd_file) as hdu:
#             asn_table = Table(hdu[1].data)
#             # print('got asn table')
#             wave = np.array(asn_table['WAVE'])
#             # print('got wave')
#             # print(wave)
#             flux_reduced = np.array(asn_table['FLUX_REDUCED'])
#             # plt.plot(wave, flux_reduced)
#             # plt.show()
#             # print('got flux reduced')
#             # print(flux_reduced)
#             # print('rv error is problem')
#             # rv_temp_uncert = np.array(asn_table['RV_ERR'][0])
#             rv_temp_uncert = 0.0
#             # print(f'template uncertainty: {rv_temp_uncert}')
#             # print('jk rv err not problem')
#             # print(f"RV err for template: {rv_temp_uncert:.5f}")

#             # print(np.shape(wave), np.shape(flux_reduced))
#             # print(f'wave: {type(wave)}')
#             # print(f'flux: {type(flux_reduced)}')
#             # print(f'rv_temp: {type(rv_temp_uncert)}')

#             ra_temp = asn_table['RA'].tolist()
#             dec_temp = asn_table['DEC'].tolist()
#             date_temp = asn_table['DATE-OBS'].tolist()
#             # exptime_temp = asn_table['EXPTIME'].tolist()
#             # print(f'ra: {type(ra_temp)}')
#             # print(f'dec: {type(dec_temp)}')
#             # print(f'date: {type(date_temp)}')
#             # print(f'exptime: {type(exptime_temp)}')

#         # print('got hd file')
#         wave_correction_tucana = np.array(wave_tucana)  # placeholder if later swap in WVCR output

#         # hard trim to common length


#         cut_mask = (wave_correction_tucana < 890)
#         cut_mask_template = (wave < 890)

#         wave_correction_tucana = wave_correction_tucana[cut_mask]
#         wave_tucana = wave_tucana[cut_mask]
#         flux_sky_sub = flux_sky_sub[cut_mask]
#         flux_err = flux_err[cut_mask]
#         wave = wave[cut_mask_template]
#         flux_reduced = flux_reduced[cut_mask_template]
#         # scaled_skyline_flux = scaled_skyline_flux[cut_mask]

#         # cut = min(len(wave_correction_tucana), len(flux_sky_sub), len(flux_err), len(wave), len(flux_reduced))
#         # wave_correction_tucana = wave_correction_tucana[:cut]
#         # wave_tucana = wave_tucana[:cut]
#         # flux_sky_sub = flux_sky_sub[:cut]
#         # flux_err = flux_err[:cut]
#         # wave = wave[:cut]
#         # flux_reduced = flux_reduced[:cut]

#         # combine calibration uncertainties
#         rv_wave_uncert = rv_wave_uncert**2 + rv_temp_uncert**2

#         if correction is True:
#             return (ra_temp, dec_temp, date_temp,
#                     wave_correction_tucana, flux_sky_sub, flux_err,
#                     wave, flux_reduced,
#                     fiber_number, priority, object_id,
#                     ra, dec, date,
#                     rv_wave_uncert)

#         return (ra_temp, dec_temp, date_temp,
#                 wave_tucana, flux_sky_sub, flux_err,
#                 wave, flux_reduced,
#                 fiber_number, priority, object_id,
#                 ra, dec, date,
#                 rv_wave_uncert)

#     except Exception as e:
#         print(f"An error occurred loading {tucana_file}: {e}")
#         return None
# =================================================================================== #


import glob

def find_old_style_file(galaxy, observation_date, object_id):
    """
    Locate the corresponding old-style reduced file (the one with
    RA/DEC/DATE-OBS in its table) for a given galaxy + observation date +
    object ID, by globbing the old 'My Reduced Data' folder structure.
    """
    old_dir = f"/Users/sashamintz/Desktop/Carnegie-USC/data/My Reduced Data/{galaxy}/{observation_date}/"
    pattern = os.path.join(old_dir, f"*OBJ={object_id}*.fits")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No old-style file found for object_id={object_id}, "
            f"galaxy={galaxy}, date={observation_date} in {old_dir}"
        )
    if len(matches) > 1:
        print(f"Warning: multiple matches for {object_id} on {observation_date}, using first: {matches[0]}")

    return matches[0]


def get_data_sky_sub(tucana_file, galaxy, correction,
                     template_file='/Users/sashamintz/Desktop/Carnegie-USC/test_willman/BOSZ/t4750_g+1.5_m-2.50_a+0.25_c+0.00.npz'):
    object_id = extract_object(tucana_file)
    # temp = '6250'
    temp = '4750'
    # temp = '5000'
    # temp = '5750'
    # logg = '4.5'
    logg = '1.5'
    # logg = '2.0'
    # logg = '3.5'
    # feh = '-2.5'
    feh = '-2.5'
    # feh = '-2.0'
    # feh = '-2.0'
    try:
        with fits.open(tucana_file) as hdu:
            data = hdu[1].data
            hdr = hdu[0].header
            good = data["mask"] == 1

            wave_tucana = data["wave"][good]
            flux_sky_sub = data["flux"][good]
            flux_err = data["sigma"][good]
            ra  = hdr["RA_OBJ"]
            dec = hdr["DEC_OBJ"]
            date = hdr["DATE-OBS"]

            rv_wave_uncert = 0.0

        with np.load(template_file) as tpl:
            wave = tpl["wave"]
            flux_reduced = tpl["flux"]

        ra_temp = 0
        dec_temp = 0
        date_temp = '0-0-0'
        rv_temp_uncert = 0.0

        wave_correction_tucana = np.array(wave_tucana)

        cut_mask = (wave_correction_tucana >= 4000) & (wave_correction_tucana <= 6500)

        wave_correction_tucana = wave_correction_tucana[cut_mask]
        wave_tucana = wave_tucana[cut_mask]
        flux_sky_sub = flux_sky_sub[cut_mask]
        flux_err = flux_err[cut_mask]

        rv_wave_uncert = rv_wave_uncert**2 + rv_temp_uncert**2

        if correction is True:
            return (ra_temp, dec_temp, date_temp,
                    wave_correction_tucana, flux_sky_sub, flux_err,
                    wave, flux_reduced,
                    object_id,
                    ra, dec, date,
                    rv_wave_uncert, temp, logg, feh)

        return (ra_temp, dec_temp, date_temp,
                wave_tucana, flux_sky_sub, flux_err,
                wave, flux_reduced,
                object_id,
                ra, dec, date,
                rv_wave_uncert, temp, logg, feh)

    except Exception as e:
        print(f"An error occurred loading {tucana_file}: {e}")
        return None

def get_priors(wave_obs, flux_obs, wave_template, flux_template):
    """
    """
    # --- limit to overlapping wavelength region ---
    min_wave = max(np.nanmin(wave_template), np.nanmin(wave_obs))
    max_wave = min(np.nanmax(wave_template), np.nanmax(wave_obs))

    mask_template = (wave_template >= min_wave) & (wave_template <= max_wave)
    mask_obs = (wave_obs >= min_wave) & (wave_obs <= max_wave)

    wave_template = np.asarray(wave_template, dtype=float)[mask_template]
    flux_template = np.asarray(flux_template, dtype=float)[mask_template]
    wave_obs = np.asarray(wave_obs, dtype=float)[mask_obs]
    flux_obs = np.asarray(flux_obs, dtype=float)[mask_obs]

    # --- reference endpoints for the normalized coordinate ---
    wave_ref_min = float(np.nanmin(wave_obs))
    wave_ref_max = float(np.nanmax(wave_obs))
    span = wave_ref_max - wave_ref_min
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    x_obs = 2.0 * (wave_obs - wave_ref_min) / span - 1.0

    # Fallback used by several early exits below.
    def _default_priors():
        b_val, c_val, d_val = 0.0, 0.0, 1.0
        e_val, f_val, g_val, h_val = 0.0, 0.0, 0.0, 0.0
        return (-10.0, 10.0, -50.0, 50.0, -10.0, 10.0,
                -50.0, 50.0, -50.0, 50.0, -50.0, 50.0, -50.0, 50.0,
                b_val, c_val, d_val, e_val, f_val, g_val, h_val, wave_ref_min, wave_ref_max)

    # --- crude continua from endpoint means ---
    num_points = 50
    num_points = min(num_points, len(wave_template) // 2, len(wave_obs) // 2)
    if num_points < 2:
        return _default_priors()

    # Template endpoint line in (wave_template, flux_template)
    y1 = np.nanmean(flux_template[:num_points])
    y2 = np.nanmean(flux_template[-num_points:])
    w1 = np.nanmean(wave_template[:num_points])
    w2 = np.nanmean(wave_template[-num_points:])
    template_slope = (y2 - y1) / (w2 - w1 + 1e-30)
    # Anchor on the endpoint MEANS, not flux_template[-1] -- a single NaN edge
    # pixel there would poison the whole array.
    template_yint = y2 - template_slope * w2
    continuum_template = template_slope * wave_template + template_yint

    # Observed endpoint line in (wave_obs, flux_obs)
    y3 = np.nanmean(flux_obs[:num_points])
    y4 = np.nanmean(flux_obs[-num_points:])
    wo1 = np.nanmean(wave_obs[:num_points])
    wo2 = np.nanmean(wave_obs[-num_points:])
    obs_slope = (y4 - y3) / (wo2 - wo1 + 1e-30)
    obs_yint = y4 - obs_slope * wo2
    continuum_obs = obs_slope * wave_obs + obs_yint

    # Ratio continuum that multiplies the template to match the observed continuum
    continuum_template_interp_func = interp1d(
        wave_template, continuum_template, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )
    continuum_template_at_obs = continuum_template_interp_func(wave_obs)

    # Mask rather than floor: np.where(|x| > 1e-12, x, 1e-12) sends NaN down the
    # false branch, silently turning it into a ~1e12 ratio that wrecks the fit.
    denom_ok = (np.abs(continuum_template_at_obs) > 1e-12) & \
               np.isfinite(continuum_template_at_obs)
    continuum_MCMC = np.full_like(wave_obs, np.nan)
    continuum_MCMC[denom_ok] = continuum_obs[denom_ok] / continuum_template_at_obs[denom_ok]

    # --- Legendre fit for the initial guess ---
    good = np.isfinite(x_obs) & np.isfinite(continuum_MCMC)
    if good.sum() < 7:
        return _default_priors()

    # legfit returns coefficients in order [P0, P1, P2], so the CONSTANT term is
    # d and the QUADRATIC term is b -- matching d*P0 + c*P1 + b*P2 in the model.
    b_val, c_val, d_val, e_val, f_val, g_val, h_val = np.polynomial.legendre.legfit(
        x_obs[good], continuum_MCMC[good], deg=6
    )

    d_val, c_val, b_val = float(d_val), float(c_val), float(b_val)
    e_val, f_val, g_val, h_val = float(e_val), float(f_val), float(g_val), float(h_val)

    if not np.all(np.isfinite([b_val, c_val, d_val, e_val, f_val, g_val, h_val])):
        return _default_priors()

    # --- priors ---
    # On [-1, 1] the Legendre coefficients are O(1), so these pads are wide
    # enough to be uninformative without being absurd. Widen if a chain pins
    # against a bound.
    pad_scale = 10000.0
    b_pad = pad_scale * (abs(b_val) + 0.1)
    c_pad = pad_scale * (abs(c_val) + 0.1)
    d_pad = pad_scale * (abs(d_val) + 0.1)
    e_pad = pad_scale * (abs(e_val) + 0.1)
    f_pad = pad_scale * (abs(f_val) + 0.1)
    g_pad = pad_scale * (abs(g_val) + 0.1)
    h_pad = pad_scale * (abs(h_val) + 0.1)

    b_min, b_max = b_val - b_pad, b_val + b_pad
    c_min, c_max = c_val - c_pad, c_val + c_pad
    d_min, d_max = d_val - d_pad, d_val + d_pad
    e_min, e_max = e_val - e_pad, e_val + e_pad
    f_min, f_max = f_val - f_pad, f_val + f_pad
    g_min, g_max = g_val - g_pad, g_val + g_pad
    h_min, h_max = h_val - h_pad, h_val + h_pad

    # Sanity clamps (all of these narrow the interval)
    b_min, b_max = max(b_min, -50.0), min(b_max, 50.0)
    c_min, c_max = max(c_min, -50.0), min(c_max, 50.0)
    d_min, d_max = max(d_min, -10.0), min(d_max, 10.0)
    e_min, e_max = max(e_min, -50.0), min(e_max, 50.0)
    f_min, f_max = max(f_min, -50.0), min(f_max, 50.0)
    g_min, g_max = max(g_min, -50.0), min(g_max, 50.0)
    h_min, h_max = max(h_min, -50.0), min(h_max, 50.0)

    # Guarantee the initial value actually sits inside its prior.
    for lo, hi, v in ((b_min, b_max, b_val), (c_min, c_max, c_val), (d_min, d_max, d_val),
                      (e_min, e_max, e_val), (f_min, f_max, f_val), (g_min, g_max, g_val),
                      (h_min, h_max, h_val)):
        if not (lo < v < hi):
            raise ValueError(
                f"Initial value {v} outside prior [{lo}, {hi}] -- loosen the clamps."
            )

    return (b_min, b_max, c_min, c_max, d_min, d_max,
            e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max,
            b_val, c_val, d_val, e_val, f_val, g_val, h_val, wave_ref_min, wave_ref_max)

def chi_squared(flux_obs, model_flux, flux_err):
    assert flux_obs.shape == model_flux.shape == flux_err.shape, "Arrays must be same length"
    residuals = flux_obs - model_flux
    chi2 = np.sum((residuals / flux_err) ** 2)
    dof = len(flux_obs) - 8
    return chi2 / dof

def run_mcmc_single(
    *,
    rv_wvcr_applied_label,
    ra_temp, dec_temp, date_temp,
    wave_obs, flux_obs, flux_err,
    wave_template, flux_template,
    object_id,
    ra, dec, date,
    correction, heco, show_plots,
    galaxy,
    pri_to_run,
    rv_wave_uncert, temp, logg, feh 
):
    """
    Runs MCMC for ONE object and writes all plots/posterior files. Returns a dict.
    """
    min_wave = max(np.min(wave_template), np.min(wave_obs))
    max_wave = min(np.max(wave_template), np.max(wave_obs))

    mask_template = (wave_template >= min_wave) & (wave_template <= max_wave)
    mask_obs = (wave_obs >= min_wave) & (wave_obs <= max_wave)

    wave_template = wave_template[mask_template]
    flux_template = flux_template[mask_template]
    wave_obs = wave_obs[mask_obs]
    flux_obs = flux_obs[mask_obs]
    flux_err = flux_err[mask_obs]

    # cutoff = min(len(wave_template), len(wave_obs))
    # wave_template = wave_template[:cutoff]
    # flux_template = flux_template[:cutoff]
    # wave_obs = wave_obs[:cutoff]
    # flux_obs = flux_obs[:cutoff]
    # flux_err = flux_err[:cutoff]

    (b_min, b_max, c_min, c_max, d_min, d_max,
     e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max,
     b_val, c_val, d_val, e_val, f_val, g_val, h_val,
     wave_ref_median, wave_ref_std) = get_priors(wave_obs, flux_obs, wave_template, flux_template)

    # hecorr_temp = get_heliocentric_correction(ra_temp, dec_temp, date_temp)
    nwalkers, ndim = 20, 8
    if galaxy == "TucIII" or galaxy == "GrusII" or galaxy == "GrusII_Verdiani":
        initial_pos = np.array([-75.0, b_val, c_val, d_val, e_val, f_val, g_val, h_val])
    elif galaxy == "SegueII":
        initial_pos = np.array([-20.0, b_val, c_val, d_val, e_val, f_val, g_val, h_val])
    else:
        initial_pos = np.array([-13.0, b_val, c_val, d_val, e_val, f_val, g_val, h_val])
    pos = [initial_pos + 1e-10 * np.random.randn(ndim) for _ in range(nwalkers)]


    move = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(wave_template, flux_template, wave_obs, flux_obs, b_min, b_max, c_min, c_max, d_min, d_max,
              e_min, e_max, f_min, f_max, g_min, g_max, h_min, h_max, wave_ref_median, wave_ref_std, flux_err),
        moves=move
    )

    max_steps = 3000
    sampler.run_mcmc(pos, max_steps, progress=True)

    print('acceptance fraction is problem for mcmc')
    print(type(np.mean(sampler.acceptance_fraction)))
    print(f"[{object_id}] MCMC done. mean_accept={np.mean(sampler.acceptance_fraction):.3f}")
    print('no acceptance fraction is not problem')

        # ------------------------------------------------------------------
    # Per-walker outlier rejection (replaces flat=True samples call)
    # ------------------------------------------------------------------
    discard = 1500

    # Keep the walker axis: shape (nsteps_after_discard, nwalkers, ndim)
    chain_3d = sampler.get_chain(discard=discard, thin=1, flat=False)
    logprob_3d = sampler.get_log_prob(discard=discard, thin=1, flat=False)  # (nsteps, nwalkers)

    # Median log-probability per walker, over the post-burn-in steps
    walker_med_logprob = np.median(logprob_3d, axis=0)  # shape (nwalkers,)

    # Robust "how far below the pack" score using MAD instead of std
    # (std gets dragged around by the very outliers you're trying to find)
    med = np.median(walker_med_logprob)
    mad = np.median(np.abs(walker_med_logprob - med))
    mad_std = 1.4826 * mad  # convert MAD to a std-equivalent for Gaussian-like data

    if mad_std > 0:
        z = (walker_med_logprob - med) / mad_std
    else:
        # all walkers identical / degenerate — nothing to reject
        z = np.zeros_like(walker_med_logprob)

    # Walkers more than ~5 MAD-sigma below the pack are considered stuck/aliased.
    # (Use a fairly strict threshold since you WANT to keep normal walker-to-walker
    # scatter — you're trying to catch walkers stuck in a totally separate mode.)
    good_walkers = np.where(z > -5)[0]
    bad_walkers = np.where(z <= -5)[0]

    if len(bad_walkers) > 0:
        print(f"[{object_id}] Rejected {len(bad_walkers)} stuck walker(s): "
              f"{bad_walkers.tolist()} (median logprob z-scores: "
              f"{np.round(z[bad_walkers], 2).tolist()})")

    if len(good_walkers) < max(2, nwalkers // 2):
        # Safety net: if "rejection" would gut the ensemble (e.g. genuinely
        # multimodal target, or threshold too aggressive), fall back to
        # keeping everything rather than silently biasing the answer.
        print(f"[{object_id}] WARNING: outlier rejection would remove too many "
              f"walkers ({len(bad_walkers)}/{nwalkers}); keeping all walkers.")
        good_walkers = np.arange(nwalkers)

    # Flatten only the surviving walkers
    samples = chain_3d[:, good_walkers, :].reshape(-1, ndim)

    # print('getting best fit params')
    best_fit_params = np.median(samples, axis=0)
    rv_best, b_best, c_best, d_best, e_best, f_best, g_best, h_best = best_fit_params

    rv_best_hecorr_all = np.nan
    if heco:
        hecorr = get_heliocentric_correction(ra, dec, date)
        print(f'heliocentric correction for {object_id}: {hecorr}')
        # hecorr_temp = get_heliocentric_correction(ra_temp, dec_temp, date_temp)
        rv_best_hecorr_all = rv_best + hecorr

        # print(f'hecorr correction: {hecorr - hecorr_temp}')

    rv_MCMC_uncert = np.std(samples[:, 0])
    # rv_wave_uncert = rv_wave_uncert[0]
    # print(f'rv wave uncert type: {type(rv_wave_uncert)}')
    # print(f'rv wave uncert: {rv_wave_uncert}')
    rv_uncert = np.sqrt(rv_MCMC_uncert**2 + rv_wave_uncert)

    # print('got rv uncert')
    adjusted_template_flux = shift_and_continuum(
        rv_best, b_best, c_best, d_best, e_best, f_best, g_best, h_best,
        wave_obs, wave_template, flux_template, wave_ref_median, wave_ref_std
    )

    # print('saving posterior')
    # Save posterior RV samples for later overplotting
    posterior_dir = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Posteriors/"
    os.makedirs(posterior_dir, exist_ok=True)
    np.save(os.path.join(posterior_dir, f"posterior_samples_{object_id}_{temp}_{logg}_{feh}.npy"), samples[:, 0])

    # === Plotting === #
    wave_obs_p = wave_obs[200:-200]
    flux_obs_p = flux_obs[200:-200]
    flux_err_p = flux_err[200:-200]
    adjusted_template_flux_p = adjusted_template_flux[200:-200]

    reduced_chi_squared = chi_squared(flux_obs_p, adjusted_template_flux_p, flux_err_p)

    # print('getting best fit vals')
    rv_best_f = rv_best
    rv_uncert_f = rv_uncert
    chi2_f = reduced_chi_squared
    rv_best_hecorr_all_f = rv_best_hecorr_all

    # print('making plots')
    # Template * continuum plot
    plt.figure()
    # print('made figure')
    plt.plot(wave_obs_p, flux_obs_p, label=f'Observed spectrum {object_id}')
    # print('plotted flux')
    # print(type(adjusted_template_flux_p))
    # print(type(rv_best_hecorr_all_f))
    # print(type(rv_uncert_f))
    # print(f'rv best hecorr type: {type(rv_best_hecorr_all_f)}')
    # print(f'rv uncert type: {type(rv_uncert_f)}')
    # print('rv uncert or rv heco problem')
    print(type(rv_best_hecorr_all_f))
    print(type(rv_uncert_f))
    plt.plot(
        wave_obs_p,
        adjusted_template_flux_p,
        label=f"Template*cont rv = {rv_best_hecorr_all_f:.4f} +/- {rv_uncert_f:.4f}"
    )
    # print('jk rv heco and rv uncert ok')
    # print('plotted tempalte flux')
    # print('made template plot')
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.legend()
    # print('labeled template plot')

    fig = plt.gcf()
    # print(f'chi2 type: {type(chi2_f)}')
    # print('chi2 the problem')
    fig.text(
        0.5, 0.02,
        f'Reduced Chi²: {chi2_f:.3f}',
        ha='center', va='center', fontsize=12, color='red',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
    )
    # print('jk chi2 not problem')

    if correction is False and heco is True:
        plt.title(f"Object {object_id} with Template")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Template + Spectrum/Template_Spectrum_with_Continuum_{object_id}_NOWVCR_HECO.png"
    elif correction is True and heco is True:
        plt.title(f"Object {object_id} with Template")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Template_Spectrum_with_Continuum_{object_id}_{temp}_{logg}_{feh}.png"
    elif correction is False and heco is False:
        plt.title(f"Object {object_id} with Template")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Template + Spectrum/Template_Spectrum_with_Continuum_{object_id}_NOWVCR_NOHECO.png"
    else:
        plt.title(f"Object {object_id} with Template")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Template + Spectrum/Template_Spectrum_with_Continuum_{object_id}_WVCR_NOHECO.png"

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    if show_plots:
        plt.show()
    else:
        plt.close()

    print('saved template plot')

    # Posterior distribution
    plt.figure(figsize=(8, 6))
    plt.hist(samples[:, 0], bins=30, density=True)
    plt.xlabel("Radial Velocity (km/s)")
    plt.ylabel("Probability Density")
    plt.grid(True)

    if correction is False and heco is True:
        plt.title(f"Posterior Distribution of RV Object {object_id}")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Posterior Distribution/Posterior_Distribution_RV_{object_id}_NOWVCR_HECO.png"
    elif correction is True and heco is True:
        plt.title(f"Posterior Distribution of RV Object {object_id}")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Posterior_Distribution_RV_{object_id}_{temp}_{logg}_{feh}.png"
    elif correction is False and heco is False:
        plt.title(f"Posterior Distribution of RV Object {object_id}")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Posterior Distribution/Posterior_Distribution_RV_{object_id}_NOWVCR_NOHECO.png"
    else:
        plt.title(f"Posterior Distribution of RV Object {object_id}")
        outpath = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Posterior Distribution/Posterior_Distribution_RV_{object_id}_WVCR_NOHECO.png"

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Trace plots
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    titles = ['Radial Velocity', 'b (x^2 coeff)', 'c (x^1 coeff)', 'd (x^0 coeff)',
              'e (P3 coeff)', 'f (P4 coeff)', 'g (P5 coeff)', 'h (P6 coeff)']
    y_labels = ['Radial Velocity (km/s)', 'b values', 'c values', 'd values',
                'e values', 'f values', 'g values', 'h values']

    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim) in newer emcee
    # You used sampler.chain (older access). We'll robustly handle both.
    if hasattr(sampler, "chain"):
        chain_for_plot = sampler.chain  # (nwalkers, nsteps, ndim)
        for i, ax in enumerate(axes.flat):
            for w in range(nwalkers):
                ax.plot(chain_for_plot[w, :, i], alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(y_labels[i])
            ax.set_title(f'Trace Plot of {titles[i]} for Object {object_id}')
            ax.grid(True)
    else:
        # chain returned as (nsteps, nwalkers, ndim)
        for i, ax in enumerate(axes.flat):
            for w in range(nwalkers):
                ax.plot(chain[:, w, i], alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(y_labels[i])
            ax.set_title(f'Trace Plot of {titles[i]} for Object {object_id}')
            ax.grid(True)

    plt.tight_layout()

    if correction is False and heco is True:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Trace Plots/Combined_Trace_Plot_Fiber_{object_id}_NOWVCR_HECO.png'
    elif correction is True and heco is True:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Combined_Trace_Plot_Fiber_{object_id}_{temp}_{logg}_{feh}.png'
    elif correction is False and heco is False:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Trace Plots/Combined_Trace_Plot_Fiber_{object_id}_NOWVCR_NOHECO.png'
    else:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Trace Plots/Combined_Trace_Plot_Fiber_{object_id}_WVCR_NOHECO.png'

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Corner plot
    fig = corner.corner(samples, labels=["rv", "b", "c", "d", "e", "f", "g", "h"],
        label_kwargs={"fontsize": 18},
        title_kwargs={"fontsize": 14},
        truths=[rv_best, b_best, c_best, d_best, e_best, f_best, g_best, h_best],
        labelpad=-0.15)
    fig.suptitle(f"Corner Plot for Object {object_id}", fontsize=16)

    if correction is False and heco is True:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Corner_Plot_RV_and_Continuum_{object_id}_NOWVCR_HECO.png'
    elif correction is True and heco is True:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Corner_Plot_RV_and_Continuum_{object_id}_{temp}_{logg}_{feh}.png'
    elif correction is False and heco is False:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Corner_Plot_RV_and_Continuum_{object_id}_NOWVCR_NOHECO.png'
    else:
        outpath = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/MCMC Images/Corner_Plot_RV_and_Continuum_{object_id}_WVCR_NOHECO.png'

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # print(f"[{day} | {object_id}] MCMC done. mean_accept={np.mean(sampler.acceptance_fraction):.3f}")

    print(object_id)
    print(rv_best_f)
    print(rv_best_hecorr_all_f)
    print(chi2_f)
    print(rv_uncert_f)
    print(b_best)
    print(c_best)
    print(d_best)
    print(e_best)
    print(f_best)
    print(g_best)
    print(h_best)

    return {
        "Objects": object_id,
        "RV": rv_best_f,
        "RV with all HECO": rv_best_hecorr_all_f,
        "Reduced Chi Squared": chi2_f,
        "RV uncertainty": rv_uncert_f,
        "b val": b_best,
        "c val": c_best,
        "d val": d_best,
        "e val": e_best,
        "f val": f_best,
        "g val": g_best,
        "h val": h_best,
        "temp": temp,
        "logg": logg,
        "feh": feh
    }

def process_one_file(args):

    (full_path, wv_correction, heco, show_plots,
     galaxy, all_or_fiber, fiber_to_run, all_or_pri, pri_to_run) = args

    if galaxy == 'TucIII':
        object_IDs = [
            "TucIII_1028775251",
            "TucIII_1028778334", 
            "TucIII_1028781097",
            "TucIII_1028785191",
            "TucIII_1028781590", 
            "TucIII_1028781494",
            "TucIII_1028780981",
            "TucIII_1028777330", 
            "6494256839524781824",
            "TucIII_1028783533", 
            "TucIII_1029839817",
            "TucIII_10299839200",
            "TucIII_1028841998",
            "TucIII_1029840750", 
            "TucIII_1029838789",
            "TucIII_1029843755",
            "TucIII_1028776526", 
            "TucIII_1028784651", 
            "TucIII_1029836578",
            "TucIII_1029840302"
            ]
    elif galaxy == "GrusII":
        object_IDs = [           
        "GrusII_947546635", "GrusII_947551707", "GrusII_947549917", "GrusII_947553809",
         "GrusII_947555289", "GrusII_947564288"
            ]
    elif galaxy == "SegueII":
        object_IDs = [
                # "SegueII_0251", 
                # "SegueII_0275", 
                # "SegueII_0213",
                # "SegueII_0189",
                # "SegueII_0193", 
                # "SegueII_0294",
                # "SegueII_0309", 
                # "SegueII_0295",
                # "SegueII_0458",
                # "SegueII_0684", 
                # "SegueII_0427", 
                # "SegueII_0537",
                # "SegueII_0543", 
                # "SegueII_0470", 
                # "SegueII_0536",
                # "SegueII_0750", 
                # "SegueII_0541",
                # "SegueII_0501",
                # "SegueII_0656",
                # "SegueII_0709",
                # "SegueII_0765",
                # "SegueII_0457",
                # "SegueII_0643",
                # "SegueII_0419",
                # "SegueII_0520",
                # "SegueII_0643",
                # "SegueII_0787",
                # "SegueII_0686",
                # "SegueII_0825",
                # "SegueII_0443",
                # "SegueII_0669"
                "SegueII_0299",
                "SegueII_0521",
                "SegueII_0449",
                "SegueII_0585"
                ]

    elif galaxy == "GrusII_Verdiani":
        object_IDs = [
            'Gr29'
        ]
    # Load
    # data = get_data(full_path, galaxy, observation_date, wv_correction)
    data = get_data_sky_sub(full_path, galaxy, wv_correction)
    if not data:
        return None

    (ra_temp, dec_temp, date_temp,
    wave_obs, flux_obs, flux_err,
    wave_template, flux_template,
    object_id,
    ra, dec, date,
    rv_wave_uncert, temp, logg, feh) = data

    # if object_id in object_IDs:

    # Filtering logic stays identical to your serial code, just moved into worker
    if all_or_fiber == 'N':
        if object_id != str(fiber_to_run):
            return None

    # If all_or_pri == 'N', you were effectively running everything (and setting pri_to_run = priority),
    # so no additional filter here.

    result = run_mcmc_single(
        rv_wvcr_applied_label="WVCR" if wv_correction else "NOWVCR",
        ra_temp=ra_temp, dec_temp=dec_temp, date_temp=date_temp,
        wave_obs=wave_obs, flux_obs=flux_obs, flux_err=flux_err,
        wave_template=wave_template, flux_template=flux_template,
        object_id=object_id,
        ra=ra, dec=dec, date=date,
        correction=wv_correction, heco=heco, show_plots=show_plots,
        galaxy=galaxy,
        pri_to_run=pri_to_run,
        rv_wave_uncert=rv_wave_uncert,
        temp = temp, logg = logg, feh = feh,
    )
    return result


# ----------------------------
# Main
# ----------------------------

def main():
    galaxy = input("What galaxy do you want to run? (e.g. TucIII, GrusII, SegueII) ").strip()
    # observation_date = input('What observation month do you want to run? ').strip()
    all_or_fiber = input("Do you want to run all fibers (Y/N)? ").strip()
    all_or_pri = input("Do you want to run a specific priority number (Y/N)? ").strip()

    fiber_to_run = None
    if all_or_fiber == 'N':
        fiber_to_run = input("Which fiber number would you like to run? ").strip()

    pri_to_run = None
    if all_or_pri == 'Y':
        pri_to_run = int(input("Which priority would you like to run? ").strip())

    # your toggles
    wv_correction = True
    heco = True
    show_plots = False

    # observation nights
    # observations = []
    # if galaxy == 'TucIII':
    #     if observation_date == 'August':
    #         observations = ['08-07-2021', '08-10-2021', '08-11-2021', '08-13-2021']
    #     elif observation_date == 'November':
    #         observations = ['11-25-2021', '11-27-2021', '11-30-2021', '12-01-2021']
    #     elif observation_date == 'May':
    #         observations = ['05-24-2022', '05-25-2022', '05-28-2022']

    # elif galaxy == 'GrusII':
    #     if observation_date == 'August':
    #         observations = ['08-07-2021', '08-08-2021', '08-10-2021', '08-13-2021']
    #     elif observation_date == 'November':
    #         observations = ['11-25-2021', '11-27-2021', '12-04-2021', '12-05-2021']
    #     elif observation_date == 'May':
    #         observations = ['05-24-2022A', '05-24-2022B', '05-25-2022A', '05-25-2022B']

    # elif galaxy == 'SegueII':
    #     if observation_date == 'SeptemberA':
    #         observations = ['09-01-2021']
    #     elif observation_date == 'December':
    #         observations = ['11-30-2021', '12-01-2021', '12-02-2021', '12-04-2021', '12-05-2021', '12-06-2021']
    #     elif observation_date == 'August':
    #         observations = ['08-26-2022', '08-28-2022A', '08-28-2022B']
    #     elif observation_date == 'SeptemberB':
    #         observations = ['09-02-2022A', '09-02-2022B', '09-02-2022C']

    # elif galaxy == "Fornax":
    #     if observation_date == 'OctoberA':
    #         # observations = ['10-04-2021', '10-05-2021A', '10-05-2021B', '10-05-2021C', '10-11-2021A', '10-11-2021B', '10-14-2021A', '10-14-2021B']
    #         observations = ['10-04-2021', '10-05-2021A','10-11-2021A', '10-14-2021A']
    #     # if observation_date == 'NovemberA':
    #     #     observations = ['11-03-2021', '11-10-2021A', '11-10-2021B' '11-10-2021C', '11-12-2021', '11-30-2021']
    #     if observation_date == 'DecemberA':
    #         # observations = ['12-05-2021A', '12-05-2021B', '12-05-2021C', '12-08-2021', '12-10-2021A', '12-10-2021B', '12-10-2021C', '12-28-2021A', '12-28-2021B', '12-29-2021A', '12-29-2021B']
    #         observations = ['12-05-2021A', '12-08-2021', '12-10-2021A', '12-28-2021A', '12-29-2021A']
    #     if observation_date == 'JanuaryA':
    #         # observations = ['01-01-2022A', '01-01-2022B', '01-02-2022A', '01-02-2022B', '01-02-2022C', '01-03-2022A', '01-03-2022B', '01-03-2022C']
    #         observations = ['01-01-2022A', '01-02-2022A', '01-03-2022A', '01-03-2022B', '01-03-2022C']
    #     # if observation_date == 'JanuaryB':
    #     #     observations = ['01-04-2022A', '01-04-2022B', '01-05-2022', '01-06-2022A', '01-06-2022B', '01-07-2022', '01-08-2022A', '01-08-2022B', '01-08-2022C']
    #     if observation_date == 'September':
    #         observations = ['09-04-2022', '09-25-2022', '09-30-2022A', '09-30-2022B']
    #     # if observation_date == 'OctoberB':
    #     #     observations = ['10-01-2022A', '10-01-2022B', '10-01-2022C', '10-22-2022']
    #     if observation_date == 'NovemberB':
    #         observations = ['11-18-2022A', '11-18-2022B']
    #     if observation_date == 'DecemberB':
    #         observations = ['12-23-2022A', '12-23-2022B', '12-26-2022', '12-28-2022', '12-29-2022']

    # elif galaxy == "LeoIV":
    #     if observation_date == "May":
    #         observations = ['05-09-2010A', '05-09-2010B', '05-09-2010C', '05-25-2010A', '05-25-2010B', '05-26-2010A', '05-26-2010B']
    #     if observation_date == "July":
    #         observations = ['07-02-2010']
    #     if observation_date == "February":
    #         observations = ['02-02-2011A', '02-02-2011B', '02-06-2011A', '02-06-2011B', '02-06-2011C', '02-06-2011D', '02-17-2011A', '02-17-2011B', ]
    #     if observation_date == "March":
    #         observations = ['03-04-2011', '03-23-2011', '03-24-2011']
    
    # elif galaxy == "LeoV":
    #     if observation_date == "July":
    #         observations = ['07-10-2011', '07-11-2011', '07-12-2011A', '07-12-2011B', '07-13-2011', '07-16-2011A', '07-16-2011B', '07-17-2011A', '07-17-2011B']
    #     if observation_date == "January":
    #         observations = ['01-11-2012A', '01-11-2012B', '01-11-2012C'] 
    #     if observation_date == "February":
    #         observations = ['02-19-2012A', '02-19-2012B', '02-28-2012']
    #     if observation_date == "March":
    #         observations = ['03-06-2012A', '03-06-2012B', '03-17-2012A', '03-17-2012B', '03-18-2012A', '03-18-2012B',]  
    # elif galaxy == "GrusII_Verdiani":
    #     if observation_date == "JulyA":
    #         observations = ['07-04-2019A', '07-04-2019B']
    #     if observation_date == "JulyB":
    #         observations = ['07-04-2021A', '07-04-2021B']
    
    # if not observations:
    #     print("No matching observation list for your inputs.")
    #     return

    # choose a sensible worker count (MCMC is CPU-heavy)
    cpu = os.cpu_count() or 2
    max_workers = max(1, cpu - 1)

    # for day in observations:
    # base_path = f'/Users/sashamintz/Desktop/Carnegie-USC/data/My Reduced Data/{galaxy}/{day}/'
    # base_path = f'/Users/sashamintz/Desktop/Carnegie-USC/data/Other/{galaxy}/Wavelength Corrected/{day}/'
    base_path = f'/Users/sashamintz/Desktop/Carnegie-USC/test_willman/Collated_Willman1_Spectra/'
    file_list = [f for f in os.listdir(base_path) if f.endswith('.fits')]

    # Build worker args for all files in this night
    tasks = []
    for fname in file_list:
        full_path = os.path.join(base_path, fname)
        tasks.append(
            (full_path, wv_correction, heco, show_plots,
                galaxy, all_or_fiber, fiber_to_run, all_or_pri, pri_to_run)
        )

    results = []
    print(f"\n=== {galaxy}: submitting {len(tasks)} files to {max_workers} workers ===\n")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_file, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out is not None:
                    results.append(out)
            except Exception as e:
                print(f"A worker failed: {e}")
                # traceback.print_exc()

    # Aggregate and write CSV
    if len(results) == 0:
        print(f"No results produced (filters may have excluded all files).")

    df = pd.DataFrame(results)

    # Stable ordering is useful when debugging
    # Fiber Numbers is a string in your parser; sort numerically if possible
    # try:
    #     df["Fiber Numbers sortkey"] = df["Fiber Numbers"].astype(int)
    #     df = df.sort_values("Fiber Numbers sortkey").drop(columns=["Fiber Numbers sortkey"])
    # except Exception:
    df = df.sort_values("Objects")

    if all_or_pri == 'Y':
        outcsv = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/Other/{galaxy}, Priority {pri_to_run}.csv"
    else:
        outcsv = f"/Users/sashamintz/Desktop/Carnegie-USC/test_willman/Other/{galaxy}_4750_1.5_-2.5.csv"

    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv, index=False)
    print(f"Wrote {len(df)} rows -> {outcsv}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} sec")