# bettermoments — Code Review

> **Status (2026-07-07):** All items below have been fixed except **M4** (30 MB
> FITS cube tracked in git — removing it requires a history rewrite, so it
> needs a decision), **m17** (the `indices` orientation heuristic, left
> documented as-is), **n4** (the 3-D-only `collapse_first/second` API
> inconsistency) and **n5** (`validate_acf.py` refinements). A pytest suite
> (81 tests, `tests/`) and a GitHub Actions workflow were added; packaging
> migrated from `setup.py` to `pyproject.toml`. Note two deliberate behaviour
> changes: `collapse_zeroth` now uses a rectangular sum (matching its
> documented uncertainty), and `-smooththreshold` is now interpreted as the
> kernel FWHM in pixels (previously it was silently a sigma).

Full-repo review (v1.10.0, 2026-07-07). Issues are grouped by severity, with
checkboxes so we can work through them. File/line references are against the
current `master` (8e33eff).

---

## Critical — data loss or always-broken code paths

- [x] **C1. Output paths built with `str.replace('.fits', ...)` can overwrite the input cube.**
  `io.py:269, 285, 302, 315, 327, 339, 357, 386`. If the input is named
  `cube.FITS`/`cube.fit`, the replace is a no-op, `new_path == args.path`, and
  `fits.writeto(..., overwrite=True)` **destroys the input cube** (e.g. with the
  mask in `_save_mask`). A `.fits` substring earlier in the path (directory named
  `run.fits/`) also mangles the output location. Fix: `os.path.splitext` +
  validate the extension, in one shared helper.

- [x] **C2. `--acf` is silently disabled on NaN-padded cubes (i.e. most CASA cubes).**
  `collapse_cube.py:112-117` with `io.py:34`. `_get_data` fills NaNs with 0.0, so
  NaN-padded pixels pass the off-source test with `var == 0`, get mapped to NaN,
  and the plain `np.mean` (not `nanmean`) makes `rho` NaN at `tau=1` → loop
  breaks → `acf = [1.0]`. The correction is a no-op exactly on the cubes that
  need it, with no warning. (The finite-spectra filter at line 96 is dead code
  from the CLI path for the same reason.) Fix: exclude `var == 0` spectra and/or
  use `np.nanmean`; warn if the ACF collapses to `[1.0]`.

- [x] **C3. `profiles.doublegauss_cont` is unusable — every call raises.**
  `profiles.py:94-97`. `free_params('doublegauss_cont')` is 7 but the function
  asserts `len(params) == 6`; with 6 params it forwards only 5 to `doublegauss`,
  which asserts 6. There is no working call path. Also: it takes
  `np.max([line, continuum])` while the docstring says the continuum is *added*.

- [x] **C4. `build_cube('gausshermite', ...)` always crashes (drops `Fnu`).**
  `profiles.py:238-244`. Unpacks `v0, dV, Fnu, h3, h4` but calls
  `gausshermite(x, v0, dV, h3, h4)` — 4 args to a 5-param model. Without the
  internal assert, `h3` would silently become the amplitude.

- [x] **C5. `estimate_p0` breaks all `_cont` model variants.**
  `mcmc_sampling.py:377-384`. Exact string comparisons (`'doublegauss' ==
  model_function`) mean `doublegauss_cont`, `gaussthick_cont`,
  `gausshermite_cont` never get their extra starting values (e.g. 4 values for
  a 5-param model). Use `startswith` / a lookup table.

- [x] **C6. `default_priors` gives `doublegauss` only 3 priors for 6 params.**
  `mcmc_sampling.py:483-492`. Checks `'multi' in model_function` (leftover from
  an old `multigauss` name); `lnprior`'s `zip` silently truncates, so the second
  component samples with **no prior at all** (improper posterior, `dVb → 0`
  NaN likelihoods).

- [x] **C7. `collapse_doublegauss` sorts components by line *width*, not peak.**
  `methods.py:481`. `p[2::6]` selects `dV`/`dVb`; the docstring promises the
  first component is the brighter one, which requires `p[4::6]` (`Fnu`).
  Whenever the fainter component is broader, primary/secondary maps are swapped.

- [x] **C8. `collapse_percentiles` uncertainties are dimensionally wrong.**
  `methods.py:332-342`. `dwgt` is a difference of *normalized* (dimensionless)
  cumulative weights but is multiplied by `rms` in intensity units, so the
  reported velocity uncertainty scales linearly with source brightness. The
  `rms` must be normalized by the total integrated weight (and the cumulative
  sum's √N noise growth considered).

- [x] **C9. No test suite and no CI.**
  No `tests/`, no GitHub Actions, no pytest config. The only verification is
  the manual `scripts/validate_acf.py`. For a package classified
  Production/Stable that just shipped new statistical machinery (ACF-corrected
  uncertainties), regressions are undetectable. Several bugs above (C3, C4, C5)
  would have been caught by a single smoke test per model. Suggest: pytest +
  small synthetic-cube fixtures + GitHub Actions matrix.

---

## Major — wrong results or broken installs in common scenarios

### Packaging / dependencies

- [x] **M1. `scipy` missing from `install_requires`.**
  `setup.py:22-29`. `io.py:5` imports `scipy.constants` at module top level (and
  `collapse_cube.py`, `methods.py` use scipy too), so `import bettermoments`
  fails on a clean install unless scipy arrives transitively.

- [x] **M2. `numpy` unpinned but `np.trapezoid` requires numpy ≥ 2.0.**
  `methods.py:79`, `mcmc_sampling.py:361` vs `setup.py:23`. Installing with
  numpy 1.x yields a runtime `AttributeError`. Pin `numpy>=2.0` (or fall back
  to `np.trapz`).

- [x] **M3. `argparse` listed as a dependency.**
  `setup.py:25`. It's stdlib; the PyPI `argparse` is an abandoned 2015 backport
  that can shadow the stdlib module. Remove.

- [ ] **M4. 30 MB FITS cube tracked in git.**
  `docs/_static/notebooks/HD135344B_13CO.fits`. Bloats every clone forever.
  Move to Zenodo/release asset with a download step in the notebook (requires a
  history rewrite or acceptance that history stays heavy).

### FITS / WCS output correctness

- [x] **M5. `RADESYS` never copied — typo `RADSYS`.**
  `io.py:252`. `header['RADSYS']` always raises `KeyError` inside the `try`,
  so the coordinate reference frame (ICRS vs FK5) is silently dropped from every
  output — the exact confusion the comment at line 245 says this prevents. The
  `'RADSYS' in new_header` check at line 255 is likewise dead.

- [x] **M6. WCS rotation matrix (`PC*_*`/`CD*_*`) and `LONPOLE`/`LATPOLE` not copied.**
  `io.py:227-231`. Only `CTYPE/CRVAL/CDELT/CRPIX/CUNIT` survive; any cube with a
  non-identity PC/CD matrix gets a silently wrong astrometric solution in all
  output maps.

- [x] **M7. `CUNIT3` ignored — km/s cubes produce maps labelled m/s.**
  `io.py:169-183`. `_read_velocity_axis` assumes m/s; a `CUNIT3='km/s'` cube
  produces v0/M1/M2 maps wrong by 1000× relative to their `BUNIT`. Also check
  the `restfreq` vs `crval3` fallback (`io.py:165`) for Hz/GHz mixing.

### CLI behavior

- [x] **M8. Module-level `warnings.filterwarnings("ignore")` for anyone importing the package.**
  `collapse_cube.py:16`. `__init__.py` imports this module, so
  `import bettermoments` disables *all* warnings in the user's session. The
  `--silent` handling at lines 417-419 is also **inverted** (`if not
  args.silent: ... ignore`) and redundant. Remove the global filter; scope any
  suppression to specific warning classes inside functions.

- [x] **M9. `-smooththreshold` units inconsistent three ways; `_get_pix_per_beam` never wired up.**
  `collapse_cube.py:383-384`, `get_threshold_mask` (279-280, 306). CLI help says
  "beam FWHM", docstring says "FWHM in pixels", code uses it as Gaussian *sigma
  in pixels* — off by 2.355× and the beam→pixel factor. `io._get_pix_per_beam`
  (203-206) exists to do the conversion but is dead code. Decide the unit,
  wire up the conversion, fix the docs.

- [x] **M10. `--acf` silently ignored for `eighth`, `ninth`, `maximum`, `percentiles`.**
  `collapse_cube.py:532-635`. Runs to completion with uncorrected uncertainties
  and no warning; the user believes their errors are ACF-corrected. Warn or
  error when `acf is not None` and the method can't use it.

### Numerics / statistics

- [x] **M11. `dM1`/`dM2` sensitivity vectors not masked to contributing channels.**
  `methods.py:140-144, 208-212`. `g` is summed over *all* channels, but masked
  channels have zero true sensitivity; for a typical threshold-masked cube the
  masked channels' `(v_i − M1)²` terms dominate and `dM1`/`dM2` are
  substantially overestimated. Zero `g` where `data == 0` (carrying
  `sign(data)` for the ACF cross-terms from the `abs(data)` weighting).

- [x] **M12. `collapse_zeroth`: trapezoidal value, rectangular uncertainty.**
  `methods.py:77-84`. `M0` uses `np.trapezoid` but `dM0 = chan·rms·√npix`
  assumes rectangular weights (end channels should carry `chan/2`). Docstring
  also shows the rectangular sum, and the second displayed equation (line ~49)
  is mislabelled `M_0` instead of `δM_0`. Make value, error, and docs
  consistent.

- [x] **M13. `wpdVb`/`wpdVr` uncertainties missing the √2 factor.**
  `methods.py:356-357`. `wpdVb = √2(wp50 − wp16)` ⇒ error is
  `√2·hypot(dwp50, dwp16)`; code returns plain `hypot` (~41% underestimate),
  and ignores the strong positive correlation between percentiles of the same
  spectrum.

- [x] **M14. `collapse_quadratic` return arity depends on `uncertainty`.**
  `quadratic.py:92-97 vs 146-150`. Returns 6 items (incl. undocumented
  curvature) with `uncertainty=None` but 4 with one; docstring says 4 always.
  Callers unpacking 4 break.

- [x] **M15. Quadratic: no `a2 == 0` guard; edge-pixel uncertainties are garbage.**
  `quadratic.py:75-76, 102-108`. Flat-topped/clipped/zero spectra → inf/NaN
  from `a1/a2`; edge pixels get `x_max`/`y_max` overridden (79-90) but their
  uncertainties from the clipped-index `gx`/`gy` are never fixed.

- [x] **M16. `gaussthick` vs `gaussthick_cont` disagree on the meaning of `Fnu`; τ=0 divides by zero.**
  `profiles.py:120-121 vs 138-141`. One normalizes by `(1 − e^{−τ})`, the other
  doesn't, so fits with/without continuum aren't comparable; the default prior
  `[0, 1e3]` and unbounded `curve_fit` both permit `tau = 0` → 0/0.

- [x] **M17. `fit_cube` continues after a caught model-import error → `NameError`.**
  `mcmc_sampling.py:183-187`. The `ValueError` is printed, not re-raised, so
  `nparams` is unbound; and `free_params` raises `KeyError` for unknown names,
  which isn't caught at all.

- [x] **M18. `mcmc='zeus'` path raises `TypeError`.**
  `mcmc_sampling.py:346-347`. `skip_initial_state_check=True` is emcee-only;
  zeus's `run_mcmc` doesn't accept it. Pass conditionally.

- [x] **M19. Non-square images crash `check_finite_errors`.**
  `methods.py:728-730`. `_interpolate_finite_errors` asserts
  `value.shape[0] == value.shape[1]`; called unconditionally from
  `collapse_cube.py:641-642`, so every rectangular cube dies with a bare
  `AssertionError`. Build the meshgrid from both axes.

---

## Minor — robustness, error handling, UX

- [x] **m1. Weight jitter `1e-10 * np.random.rand(...)` makes M1/M2 non-deterministic.**
  `methods.py:137, 200`. No seed; use a deterministic epsilon or mask
  `np.average` explicitly.

- [x] **m2. `-method` validated last; unknown method message unhelpful.**
  `collapse_cube.py:634-635`. A typo wastes the full load/mask/smooth/ACF
  pipeline before failing with "Unknown method." Use `choices=` in
  `add_argument` (also fixes `collapse_method_products` returning `None` →
  `AttributeError` in `io.py:380`; raise there too — methods.py:696-697).

- [x] **m3. `-clip` with 0 values → bare `IndexError`; 3+ values silently dropped.**
  `collapse_cube.py:296-297, 322` (`nargs='*'`). Constrain/validate nargs.

- [x] **m4. Channel-mask validation via bare `assert`s; single-channel selection rejected.**
  `collapse_cube.py:230-232`. Message-less `AssertionError`s (and gone under
  `python -O`); `firstchannel == lastchannel` should be legal. Also the
  `-lastchannel` error message at 410-411 contradicts the `< 1` check.

- [x] **m5. `estimate_RMS` has no `2N ≥ nchan` guard.**
  `collapse_cube.py:21-38`. Short cubes double-count line channels in the noise
  estimate, silently inflating the RMS (and the ACF off-source threshold).

- [x] **m6. Missing `BUNIT` → raw `KeyError` after all computation.**
  `io.py:45, 226`. Fail fast at load time or fall back to `''` with a warning.

- [x] **m7. `_collapse_beamtable`: leaked file handle, docstring/code mismatch, silent wrong fallback.**
  `io.py:186-200`. `fits.open(path)[1]` never closed; docstring says median but
  code takes max; the no-beam fallback writes the *pixel size* as `BMAJ` into
  every output header with no warning.

- [x] **m8. Non-standard `RESTFREQ` key and invented `0.0` value.**
  `io.py:232-238`. FITS standard is `RESTFRQ`; writing `0.0` when unknown
  looks like a real (absurd) rest frequency downstream.

- [x] **m9. Header/beamtable re-read from disk per output plane.**
  `io.py:385`. `_get_bunits`/`_write_header` inside the per-moment loop — ~36
  header reads for `doublegauss`. Hoist out.

- [x] **m10. Debug/mask outputs ignore `-outname`.**
  `io.py:279-284` etc. Always written next to the input cube; breaks on
  read-only data directories.

- [x] **m11. Stokes handling fragile.**
  `io.py:31-34`. Out-of-range `-stokes` → raw `IndexError`; Stokes assumed to
  be NAXIS4 (a Stokes-on-axis-3 cube silently selects a spectral plane); NaN→0
  fill is silent (see C2).

- [x] **m12. `collapse_width` and `dM2` divide-by-zero guards.**
  `methods.py:640-641` (`Fnu`, `M0`), `methods.py:208` (`2·M2`). Noise pixels
  produce inf + runtime warnings instead of NaN.

- [x] **m13. `doublegauss` returns the max of the two Gaussians, not their sum.**
  `profiles.py:77`. Contradicts docstrings ("two gaussian components",
  "added"); if max is intentional (optically-thick overlapping layers),
  document it — note it's non-differentiable, which hurts `curve_fit`.

- [x] **m14. `fit_spectrum(niter=0)` → `NameError`; `optimize_p0` misses `ValueError`.**
  `mcmc_sampling.py:262-292`. Validate `niter ≥ 1`; catch `curve_fit`'s
  `ValueError` on NaN input.

- [x] **m15. `_dx_prior` allows `dV = 0`.**
  `mcmc_sampling.py:441-443`. Every model divides by `params[1]`; use a small
  positive floor.

- [x] **m16. `~np.isfinite(lnp)` idiom.**
  `mcmc_sampling.py:141`. Works only for numpy scalars; use `not np.isfinite(lnp)`.

- [ ] **m17. `indices` orientation heuristic ambiguous.**
  `mcmc_sampling.py:191`. `(2, 2)` input is undecidable and `(2, N)` lists get
  silently mis-transposed; require/document `(N, 2)`.

- [x] **m18. Dead code sweep.**
  `mcmc_sampling.py:519-527` `verify_fits` never called (and `empty` can be
  unbound); `diagnostic_plots`'s `mcmc` arg unused (530);
  `quadratic.py:79-90` scalar `else` branch unreachable; `io._get_pix_per_beam`
  never called (see M9); `collapse_cube.py:485` `N=args.noisechannels` arg is
  dead (rms always non-None there); `io.py:145` `'mask': 'bool'` BUNIT never
  used (masks carry `Jy/beam`).

- [x] **m19. NaN counted as valid data.**
  `methods.py:769` `_get_finite_pixels` tests `data != 0.0` (NaN passes); same
  issue inflates `npix` in `collapse_zeroth/first/second`. Redundant/duplicated
  masking in `collapse_eighth`/`collapse_ninth` (243-248, 267-272; `np.max`
  computed three times).

- [x] **m20. `from .methods import *` / `from .io import *` without `__all__`.**
  `__init__.py:4-5`. Re-exports `np` as `bettermoments.np`, etc. Define
  `__all__` in each module.

### Packaging / docs (minor)

- [x] **m21. Migrate metadata from `setup.py` to `pyproject.toml [project]`.**
  `pyproject.toml` has only the build-system table; setuptools is deprecating
  setup.py-only metadata. Do M1-M3 as part of this. Add `python_requires`,
  single-source the version (currently duplicated in `setup.py:13` and
  `__init__.py:7`), and fix `license="LICENSE.md"` → `license="MIT"` +
  `license_files`.

- [x] **m22. `docs/index.rst` toctree omits `user/faq` and `Cookbook_2`.**
  `docs/index.rst:34-43` — both build as orphaned, unreachable pages.

- [x] **m23. Undocumented CLI flags.**
  `-outname`, `-processes`, `-stokes`, `--returnmodel`, `--silent`, `--debug`
  (`collapse_cube.py:373-403`) appear nowhere in `docs/user/command_line.rst`.
  (The `--acf` feature itself is well documented.)

- [x] **m24. README doesn't mention the headline v1.10.0 ACF feature.**

- [x] **m25. Docs build cleanups.**
  `docs/conf.py:16` `os.path.dirname(__name__)` should be `__file__`; no
  `version`/`release` set; `docs/requirements.txt` lists `sphinx-rtd-theme`
  twice and the pip `pandoc` package is not the pandoc binary nbsphinx needs —
  verify the RTD notebook build. Prefer installing the package on RTD over
  `autodoc_mock_imports`.

- [x] **m26. Tracked Jupyter checkpoint.**
  `docs/_static/notebooks/.ipynb_checkpoints/...` is in git; remove and add
  `.ipynb_checkpoints/` to `.gitignore`.

- [x] **m27. Add `CITATION.cff`.**
  Citation info (Zenodo DOI, ASCL, BibTeX) is internally consistent, but a
  `CITATION.cff` would give GitHub's "Cite this repository" box. Also
  reconcile copyright year/name between `LICENSE.md:3` (2018, "Rich Teague")
  and `docs/conf.py:21` (2020-2026, "Richard Teague").

---

## Nits

- [x] **n1. Docstring copy-paste and typos in `methods.py`.**
  `collapse_gausshermite` docstring describes gaussthick (425-427); documented
  return orders for `collapse_gausshermite`/`collapse_doublegauss` contradict
  the actual `(v0, dV, Fnu)` order in `collapse_method_products` (686-692);
  typos "Maksed", "intesity", "chanenl", "Optioinal", "Optioanl". Stray `"""`
  glued to an assert message at `methods.py:768`.
- [x] **n2. `--nooverwrite` uses `action='store_false'`** so `nooverwrite is
  True` means "overwrite" (`collapse_cube.py:397-398`); use `dest='overwrite'`.
- [x] **n3. `io.py:49-52` duplicated comment** — first block should say
  `# method='zeroth'`.
- [ ] **n4. `collapse_first/second` hard-require 3-D input** (`[:, None, None]`)
  while `collapse_zeroth`/quadratic accept any ndim — inconsistent API.
- [ ] **n5. `scripts/validate_acf.py`**: analytic predictions taken from a single
  extra noise realisation (jitter near the 0.9/1.1 PASS boundaries) and the
  `acf=None` diagonal column is printed but never checked.
- [x] **n6. Trailing whitespace** `collapse_cube.py:645`; `pyproject.toml`
  missing trailing newline.

---

## Verified non-issues

- The quadratic sensitivity vectors `gx`/`gy` check out analytically.
- Gauss-Hermite normalization matches van der Marel & Franx (1993) given the
  `dV = √2σ` convention.
- `_propagate_covariance` and the 3×3 ACF block in `quadratic.py` are correct
  given `build_spectral_covariance`'s eigenvalue-floored Toeplitz construction
  (modulo M11's `g`-masking).
- `build/`, `dist/`, `*.egg-info/`, `__pycache__` exist on disk but are not
  tracked; `.gitignore` covers them.
- `setup.py` and `__init__.py` versions currently agree (1.10.0).
- `emcee`/`zeus` are imported lazily inside `fit_spectrum`, so docs mocking is fine.
- `.readthedocs.yaml` is a valid v2 config.

---

## Suggested order of attack

1. **Safety first:** C1 (path overwrite), M8 (global warning suppression), M5-M7 (WCS/header correctness) — small, self-contained fixes.
2. **Bootstrap tests (C9):** pytest + synthetic-cube fixtures + GitHub Actions. Even smoke tests per method/model would have caught C3-C5.
3. **Statistics fixes:** C2, C7, C8, M11-M13 — these change scientific output, so land them with validation (extend `validate_acf.py` into proper MC tests).
4. **Model/MCMC fixes:** C3-C6, M16-M18.
5. **Packaging (M1-M4, m21):** one PR migrating to `pyproject.toml` with correct deps.
6. **CLI/UX and minor robustness (M9, M10, m1-m20), then docs (m22-m27) and nits.**
