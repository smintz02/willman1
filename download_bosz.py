import os
import re
import urllib.request
import urllib.error
import numpy as np


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BOSZ_BASE = "https://archive.stsci.edu/hlsps/bosz/bosz2024/"
RESOLUTION = "r5000"
WAVE_GRID = f"wavelength_grids/bosz2024_wave_{RESOLUTION}.txt"

TEMPLATE_DIR = "/Users/sashamintz/Desktop/Carnegie-USC/test_willman/BOSZ/"
RAW_DIR = os.path.join(TEMPLATE_DIR, "raw")   # untouched downloads kept here

WAVE_MIN, WAVE_MAX = 4000.0, 6500.0   # Angstrom, same cut as the MCMC

# (Teff, logg, [M/H], [alpha/M], [C/M])
TEMPLATES = [
    (4750, 1.5, -2.50, 0.25, 0.00),   # cool metal-poor RGB
    (5000, 2.0, -2.00, 0.25, 0.00),   # warmer metal-poor RGB
    (5750, 3.5, -2.00, 0.25, 0.00),   # subgiant / turnoff
    (6250, 4.5, -2.50, 0.25, 0.00),   # metal-poor dwarf (current template; foreground stars)
    # (4750, 1.5, -2.50, 0.25, 0.50), # carbon-enhanced giant
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_params(teff, logg, mh, alpha, carbon):
    """e.g. t4750_g+1.5_m-2.50_a+0.25_c+0.00 (BOSZ 2024 naming)."""
    return f"t{teff:04d}_g{logg:+.1f}_m{mh:+.2f}_a{alpha:+.2f}_c{carbon:+.2f}"


def list_directory(url):
    """Return the set of BOSZ model filenames in an archive folder, or an empty set."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            html = r.read().decode("utf-8", "replace")
        return set(re.findall(r'href="(bosz2024_[^"]+\.txt\.gz)"', html))
    except Exception as e:
        print(f"  (could not read directory listing, will guess filenames: {e})")
        return set()


def candidate_names(params, logg, listing):
    """Filenames to try for one model, best guess first."""
    giant = logg < 3.5
    prefixes = ("ms", "mp", "ap") if giant else ("mp", "ms", "ap")
    preferred_v = 2 if giant else 1

    pattern = re.compile(
        rf"bosz2024_([a-z]{{2}})_{re.escape(params)}_v(\d+)_{RESOLUTION}_resam\.txt\.gz"
    )
    found = [m for m in (pattern.fullmatch(f) for f in listing) if m]
    if found:
        found.sort(key=lambda m: (
            prefixes.index(m.group(1)) if m.group(1) in prefixes else len(prefixes),
            abs(int(m.group(2)) - preferred_v),
        ))
        return [m.group(0) for m in found]

    vturbs = sorted((0, 1, 2, 4), key=lambda v: abs(v - preferred_v))
    return [f"bosz2024_{p}_{params}_v{v}_{RESOLUTION}_resam.txt.gz"
            for p in prefixes for v in vturbs]


def download(url, dest):
    """Download url -> dest (skips if dest already exists)."""
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def fetch_model(teff, logg, mh, alpha, carbon):
    """Download one model; returns (local_path, remote_name) or (None, None)."""
    params = fmt_params(teff, logg, mh, alpha, carbon)
    folder = f"{BOSZ_BASE}{RESOLUTION}/m{mh:+.2f}/"

    for name in candidate_names(params, logg, list_directory(folder)):
        try:
            path = download(folder + name, os.path.join(RAW_DIR, name))
            return path, name
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    return None, None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Wavelength grid...")
    wave_path = download(BOSZ_BASE + WAVE_GRID,
                         os.path.join(RAW_DIR, os.path.basename(WAVE_GRID)))
    wave_full = np.loadtxt(wave_path)          # Angstrom
    wmask = (wave_full >= WAVE_MIN) & (wave_full <= WAVE_MAX)

    for teff, logg, mh, alpha, carbon in TEMPLATES:
        label = fmt_params(teff, logg, mh, alpha, carbon)
        out = os.path.join(TEMPLATE_DIR, label + ".npz")
        print(f"\n{label}")

        if os.path.exists(out):
            print("  already prepared, skipping")
            continue

        raw_path, remote_name = fetch_model(teff, logg, mh, alpha, carbon)
        if raw_path is None:
            print("  WARNING: no matching file found on the archive. "
                  "Check the grid values / filename in "
                  f"{BOSZ_BASE}{RESOLUTION}/m{mh:+.2f}/")
            continue
        print(f"  downloaded {remote_name}")

        flux, cont = np.loadtxt(raw_path, unpack=True)
        if len(flux) != len(wave_full):
            raise ValueError(f"{remote_name}: {len(flux)} flux points but "
                             f"{len(wave_full)} wavelength points")

        wave = wave_full[wmask]
        flux_norm = flux[wmask] / cont[wmask]

        np.savez(out, wave=wave, flux=flux_norm,
                 teff=teff, logg=logg, mh=mh, alpha=alpha, carbon=carbon,
                 source=remote_name)
        print(f"  saved -> {out}  ({len(wave)} pixels, "
              f"{wave[0]:.1f}-{wave[-1]:.1f} A)")

    print("\nDone.")


if __name__ == "__main__":
    main()