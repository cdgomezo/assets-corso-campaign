from xarray import open_dataset
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from lumia.Tools.regions import region
from datetime import datetime
from lumia.obsdb import obsdb
from numpy import corrcoef, where, sqrt, array
from pandas import read_csv
from pandas.tseries.frequencies import to_offset
from h5py import File

import xarray as xr
import os

def toTimeUnits(path, cat, time_span, reg='Study Domain'):

    # Check if the file exists
    step = path.split('/')[-1].split('.')[1]
    out_path = os.path.join(os.path.dirname(path), f'{step}_{cat.replace("/", "_")}_{time_span}_{reg.replace(" ", "_")}.nc')
    if not os.path.isfile(out_path):
        # Get the land mask
        if reg != 'Study Domain':
            land_mask = getRegion(reg)
        else:
            land_mask = 1
        area_05x05_EU = region(lat0=33, lat1=73, lon0=-15, lon1=35, dlat=0.5, dlon=0.5).area
        time_spans = {'Annual': ('Y', 'PgC/year', 1000),
                    'Monthly': ('MS', 'TgC/day', [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                    'Weekly': ('7D', 'TgC/day', 7),
                    'Daily': ('D', 'TgC/day', 1),
                    'Hourly': ('H', 'TgC/hour', 1)}
        time_span_code, unit, time_span_divisor = time_spans[time_span]
        ds = open_dataset(path, group=cat)
        time_ds = [datetime(*x) for x in ds.times_start.values]
        ds = xr.Dataset(data_vars={'emis': (['time', 'lat', 'lon'], ds.emis.values)}, coords={'time': time_ds, 'lat': ds.lats.values, 'lon': ds.lons.values})
        try:
            ds['emis'] = ds.emis * 12 * 1e-18 * area_05x05_EU * 3600 * land_mask.values
        except:
            ds['emis'] = ds.emis * 12 * 1e-18 * area_05x05_EU * 3600 
        ds = ds.resample(time=time_span_code, label='left', closed='left').sum(dim=['time', 'lat', 'lon']) 
        if time_span != 'Annual':
            ds = ds.sel(time=slice('2018-01-01', '2018-12-31'))
        ds['emis'] = ds.emis / time_span_divisor
        ds['emis'].attrs['units'] = unit
        ds.to_netcdf(out_path)
    else:
        ds = open_dataset(out_path)
        unit = ds.emis.units

    return ds, unit

def toTimeUnitsGrid(path, cat, time_span, reg='EU'):

    # Check if the file exists
    step = path.split('/')[-1].split('.')[1]
    out_path = os.path.join(os.path.dirname(path), f'{step}_{cat.replace("/", "_")}_{time_span}_{reg.replace(" ", "_")}_grid.nc')
    if not os.path.isfile(out_path):
        # Get the land mask
        if reg != 'EU':
            land_mask = getRegion(reg)
        else:
            land_mask = 1
        area_05x05_EU = region(lat0=33, lat1=73, lon0=-15, lon1=35, dlat=0.5, dlon=0.5).area
        time_spans = {'Annual': ('Y', 'gC m^-2 day^-1', 1000),
                      'Triannual': ('4MS', 'gC m^-2 day^-1', 1000/3),
                    'Monthly': ('MS', 'TgC/day', [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]),
                    'Weekly': ('7D', 'TgC/day', 7),
                    'Daily': ('D', 'TgC/day', 1),
                    'Hourly': ('H', 'TgC/hour', 1)}
        time_span_code, unit, time_span_divisor = time_spans[time_span]
        ds = open_dataset(path, group=cat)
        time_ds = [datetime(*x) for x in ds.times_start.values]
        ds = xr.Dataset(data_vars={'emis': (['time', 'lat', 'lon'], ds.emis.values)}, coords={'time': time_ds, 'lat': ds.lats.values, 'lon': ds.lons.values})
        try:
            ds['emis'] = ds.emis * 12 * 1e-6 * 3600 * land_mask.values
        except:
            ds['emis'] = ds.emis * 12 * 1e-6 * 3600 
        ds = ds.resample(time=time_span_code, label='left', closed='left').sum(dim=['time'])#, 'lat', 'lon']) 
        ds['emis'] = ds.emis / 365
        ds['emis'].attrs['units'] = unit
        ds.to_netcdf(out_path)
    else:
        ds = open_dataset(out_path)
        unit = ds.emis.units

    return ds, unit

def calcTotalUncertainty(Ch, data):

    unitconv = 12.e-21
    nt = len(data.itime.unique())

    errtot = []
    for it in range(nt):
        sigmas = data.loc[data.itime == it].prior_uncertainty.values * unitconv
        err = (Ch * sigmas[None, :] * sigmas[:, None]).sum()
        errtot.append(sqrt(err))

    return array(errtot)

def plot_scatter_cat(cat, alias, time_span, alias_dict):

    f, ax = subplots(1, 2, figsize=(12, 8))

    # for alias in experiments:

    ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span)
    ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apri.nc', cat, time_span)
    ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apos.nc', cat, time_span)

    ax[0].scatter(ds_truth.emis, ds_apri.emis, label=alias)
    ax[1].scatter(ds_truth.emis, ds_apos.emis, label=alias)

    r_apri = corrcoef(ds_truth.emis, ds_apri.emis)[0,1]
    r_apos = corrcoef(ds_truth.emis, ds_apos.emis)[0,1]

    r2_apri = r_apri**2
    r2_apos = r_apos**2

    ax[0].text(0.05, 0.95, f'R$^2$ apri = {r2_apri:.2f}', transform=ax[0].transAxes, va='top', size=12)
    ax[1].text(0.05, 0.95, f'R$^2$ apos = {r2_apos:.2f}', transform=ax[1].transAxes, va='top', size=12)

    ax[0].set_xlabel('Truth')
    ax[0].set_ylabel('Prior')
    ax[1].set_xlabel('Truth')
    ax[1].set_ylabel('Posterior')


def plot_flux_timeseries(category, experiments, time_span, sec_axis, alias_dict, step, error_bars=False):

    f, ax = subplots(2, 1, figsize=(20, 8))

    if sec_axis:
        ax2 = ax[0].twinx()

    for i, cat in enumerate(category):
        if 'Truth' in step:
            ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span)
            if i == 0:
                ds_truth.emis.plot(ax=ax[0], label=f'Truth {cat}', marker='o', linestyle='-')
            else:
                if sec_axis:
                    # ax2 = ax[0].twinx()
                    ds_truth.emis.plot(ax=ax2, label=f'Truth {cat}', marker='o', linestyle='--')
                else:
                    ds_truth.emis.plot(ax=ax[0], label=f'Truth {cat}', marker='o', linestyle='--')
        for j, alias in enumerate(experiments):
            if 'Prior' in step:
                ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apri.nc', cat, time_span)
            if 'Posterior' in step:
                ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apos.nc', cat, time_span)
            if i == 0:
                if 'Prior' in step:
                    ds_apri.emis.plot(ax=ax[0], label=f'{alias} Prior {cat}', marker='s')
                if 'Posterior' in step:
                    ds_apos.emis.plot(ax=ax[0], label=f'{alias} Posterior {cat}', marker='^')
                ax[0].legend(loc='upper left')
                ax[0].set_ylabel(unit)
            else:
                if sec_axis:
                    if 'Prior' in step:
                        ds_apri.emis.plot(ax=ax2, label=f'{alias} Prior {cat}', marker='s', linestyle='--')
                    if 'Posterior' in step:
                        ds_apos.emis.plot(ax=ax2, label=f'{alias} Posterior {cat}', marker='^', linestyle='--')
                    ax2.legend(loc='upper right')
                    ax2.set_ylabel(unit)
                else:
                    if 'Prior' in step:
                        ds_apri.emis.plot(ax=ax[0], label=f'{alias} Prior {cat}', marker='s', linestyle='--')
                    if 'Posterior' in step:
                        ds_apos.emis.plot(ax=ax[0], label=f'{alias} Posterior {cat}', marker='^', linestyle='--')
                    ax[0].legend(loc='upper left')
                    ax[0].set_ylabel(unit)
    
    categories = ['co2/fossil', 'co2/biosphere', 'co2/ocean']

    if 'Truth' in step:
        for i, cat in enumerate(categories):

            ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span)

            if i == 0:
                ds_truth_sum = ds_truth
            else:
                ds_truth_sum += ds_truth

        ds_truth_sum.emis.plot(ax=ax[1], label=f'Truth', marker='o', linestyle='-')

    for alias in experiments:
        for i, cat in enumerate(categories):
            
            if 'Prior' in step:
                ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apri.nc', cat, time_span)
            if 'Posterior' in step:
                ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apos.nc', cat, time_span)

            if i == 0:
                if 'Prior' in step:
                    ds_apri_sum = ds_apri
                if 'Posterior' in step:
                    ds_apos_sum = ds_apos
            else:
                if 'Prior' in step:
                    ds_apri_sum += ds_apri
                if 'Posterior' in step:
                    ds_apos_sum += ds_apos
        if 'Prior' in step:    
            ds_apri_sum.emis.plot(ax=ax[1], label=f'{alias} Prior', marker='s')
        if 'Posterior' in step:
            ds_apos_sum.emis.plot(ax=ax[1], label=f'{alias} Posterior', marker='v')

    ax[1].legend()
    ax[1].set_title('Total CO$_2$')
    ax[1].set_ylabel(unit)

    f.tight_layout()


def plot_conc_timeseries(station, experiments, all_sites, alias_dict):

    f, ax = subplots(2, 2, figsize=(20, 8), width_ratios=[2, 1])

    for i, exp in enumerate(experiments):

        db_apri = obsdb(f'/home/carlos/Project/LUMIA/results/{alias_dict[exp]}/observations.apri.tar.gz').observations
        db_apos = obsdb(f'/home/carlos/Project/LUMIA/results/{alias_dict[exp]}/observations.apos.tar.gz').observations

        for tr in ['co2', 'c14']:

            if tr == 'co2':
                unit = 'CO$_2$ [ppm]'
                
                if i == 0:
                    if all_sites:
                        db_apri.loc[(db_apri['tracer'] == tr)].plot(ax=ax[0,0], y='obs', x='time', label=f'Truth', marker='o', linewidth=0)
                        # ax[0,0].fill_between(db_apri.loc[(db_apri['tracer'] == tr)].time, db_apri.loc[(db_apri['tracer'] == tr)].obs - db_apri.loc[(db_apri['tracer'] == tr)].err, db_apri.loc[(db_apri['tracer'] == tr)].obs + db_apri.loc[(db_apri['tracer'] == tr)].err, alpha=0.5, zorder=20)
                    else:
                        db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].plot(ax=ax[0,0], y='obs', x='time', label=f'Truth', marker='o', linewidth=0)
                        # ax[0,0].fill_between(db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].time, db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].obs - db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].err, db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].obs + db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].err, alpha=0.5, zorder=20)
                
                if all_sites:
                    db_apri.loc[(db_apri['tracer'] == tr)].plot(ax=ax[0,0], y='mix_apri', x='time', label=f'{exp} Prior', marker='s', alpha=0.5, linewidth=1)
                    db_apos.loc[(db_apri['tracer'] == tr)].plot(ax=ax[0,0], y='mix_apos', x='time', label=f'{exp} Posterior', marker='v', alpha=0.5, linewidth=1)
                    ax[0,0].set_title(f'{tr}')
                    ax[0,0].set_ylabel(unit)
                    ax[0,0].legend()

                    r_apri = corrcoef(db_apri.obs.loc[db_apri.tracer == 'co2'], db_apri.mix_apri.loc[db_apri.tracer == 'co2'])[0,1]
                    r_apos = corrcoef(db_apos.obs.loc[db_apos.tracer == 'co2'], db_apos.mix_apos.loc[db_apos.tracer == 'co2'])[0,1]

                    r2_apri = r_apri**2
                    r2_apos = r_apos**2

                    ax[0,0].text(0.05, 0.95 - (i*1), f'R$^2$ apri {exp} = {r2_apri:.2f}', transform=ax[0,0].transAxes, va='top', size=12)
                    ax[0,0].text(0.05, 0.90 - (i*1), f'R$^2$ apos {exp} = {r2_apos:.2f}', transform=ax[0,0].transAxes, va='top', size=12)

                    ax[0,1].hist(db_apos.obs.loc[db_apos.tracer == 'co2'] - db_apos.mix_apri.loc[db_apos.tracer == 'co2'], bins=1000, label=f'{exp} Prior', density=True, alpha=0.5)
                    ax[0,1].hist(db_apos.obs.loc[db_apos.tracer == 'co2'] - db_apos.mix_apos.loc[db_apos.tracer == 'co2'], bins=1000, label=f'{exp} Posterior', density=True, alpha=0.5)
                    ax[0,1].set_title(f'{tr} mismatch')
                    ax[0,1].set_xlim(-5, 5)
                    ax[0,1].legend()

                else:
                    db_apri.loc[(db_apri['site'] == station) & (db_apri['tracer'] == tr)].plot(ax=ax[0,0], y='mix_apri', x='time', label=f'{exp} Prior', marker='s', alpha=0.5, linewidth=1)
                    db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].plot(ax=ax[0,0], y='mix_apos', x='time', label=f'{exp} Posterior', marker='v', alpha=0.5, linewidth=1)

                    ax[0,0].set_title(f'{station} {tr}')
                    ax[0,0].set_ylabel(unit)
                    ax[0,0].legend()

                    r_apri = corrcoef(db_apri.obs.loc[(db_apri.site == station) & (db_apri.tracer == 'co2')], db_apri.mix_apri.loc[(db_apri.site == station) & (db_apri.tracer == 'co2')])[0,1]
                    r_apos = corrcoef(db_apos.obs.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')], db_apos.mix_apos.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')])[0,1]

                    r2_apri = r_apri**2
                    r2_apos = r_apos**2

                    ax[0,0].text(0.05, 0.95 - (i*1), f'R$^2$ apri {exp} = {r2_apri:.2f}', transform=ax[0,0].transAxes, va='top', size=12)
                    ax[0,0].text(0.05, 0.90 - (i*1), f'R$^2$ apos {exp} = {r2_apos:.2f}', transform=ax[0,0].transAxes, va='top', size=12)

                    ax[0,1].hist(db_apos.obs.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')] - db_apos.mix_apri.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')], bins=1000, label=f'{exp} Prior', density=True, alpha=0.5)
                    ax[0,1].hist(db_apos.obs.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')] - db_apos.mix_apos.loc[(db_apos.site == station) & (db_apos.tracer == 'co2')], bins=1000, label=f'{exp} Posterior', density=True, alpha=0.5)
                    ax[0,1].set_title(f'{station} {tr} mismatch')
                    ax[0,1].set_xlim(-5, 5)
                    ax[0,1].legend()

            else:

                try:
                    unit = 'C14 [‰]'

                    db_apri['obs_permil'] = (db_apri.obs / db_apri.mix_fg_CO2) * 1e3
                    db_apri['apri_permil'] = (db_apri.mix_apri / (db_apri.filter(regex='mix_co2').sum(axis=1) + db_apri.mix_bg_CO2)) * 1e3

                    db_apos['apos_permil'] = (db_apos.mix_apos / (db_apos.filter(regex='mix_co2').sum(axis=1) + db_apos.mix_bg_CO2)) * 1e3

                    if all_sites:
                        if i == 0:
                            db_apri.loc[(db_apri['tracer'] == tr)].plot(ax=ax[1,0], y='obs_permil', x='time', label=f'Truth', marker='o', linewidth=0)

                        db_apri.loc[(db_apri['tracer'] == tr)].plot(ax=ax[1,0], y='apri_permil', x='time', label=f'{exp} Prior', marker='s', alpha=0.5, linewidth=1)
                        db_apos.loc[(db_apos['tracer'] == tr)].plot(ax=ax[1,0], y='apos_permil', x='time', label=f'{exp} Posterior', marker='v', alpha=0.5, linewidth=1)
                        ax[1,0].set_title(f'{tr}')
                        ax[1,0].set_ylabel(unit)
                        ax[1,0].legend()

                        r_apri = corrcoef(db_apri.obs_permil.loc[db_apri.tracer == 'c14'], db_apri.apri_permil.loc[db_apri.tracer == 'c14'])[0,1]
                        r_apos = corrcoef(db_apri.obs_permil.loc[db_apri.tracer == 'c14'], db_apos.apos_permil.loc[db_apos.tracer == 'c14'])[0,1]

                        r2_apri = r_apri**2
                        r2_apos = r_apos**2

                        ax[1,0].text(0.05, 0.95 - (i*1), f'R$^2$ apri {exp} = {r2_apri:.2f}', transform=ax[1,0].transAxes, va='top', size=12)
                        ax[1,0].text(0.05, 0.90 - (i*1), f'R$^2$ apos {exp} = {r2_apos:.2f}', transform=ax[1,0].transAxes, va='top', size=12)

                        ax[1,1].hist(db_apri.obs_permil.loc[db_apri.tracer == 'c14'] - db_apri.apri_permil.loc[db_apri.tracer == 'c14'], bins=10, label=f'{exp} Prior', density=True, alpha=0.5)
                        ax[1,1].hist(db_apri.obs_permil.loc[db_apri.tracer == 'c14'] - db_apos.apos_permil.loc[db_apos.tracer == 'c14'], bins=10, label=f'{exp} Posterior', density=True, alpha=0.5)
                        ax[1,1].set_title(f'{tr} mismatch')
                        ax[1,1].set_xlim(-10, 10)
                        ax[1,1].legend()

                    else:
                            
                        if i == 0:
                            db_apri.loc[(db_apri['site'] == station) & (db_apri['tracer'] == tr)].plot(ax=ax[1,0], y='obs_permil', x='time', label=f'Truth', marker='o', linewidth=0)
                        
                        db_apri.loc[(db_apri['site'] == station) & (db_apri['tracer'] == tr)].plot(ax=ax[1,0], y='apri_permil', x='time', label=f'{exp} Prior', marker='s', alpha=0.5, linewidth=1)
                        db_apos.loc[(db_apos['site'] == station) & (db_apos['tracer'] == tr)].plot(ax=ax[1,0], y='apos_permil', x='time', label=f'{exp} Posterior', marker='v', alpha=0.5, linewidth=1)

                        ax[1,0].set_title(f'{station} {tr}')
                        ax[1,0].set_ylabel(unit)
                        ax[1,0].legend()

                        r_apri = corrcoef(db_apri.obs_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')], db_apri.apri_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')])[0,1]
                        r_apos = corrcoef(db_apri.obs_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')], db_apos.apos_permil.loc[(db_apos.site == station) & (db_apos.tracer == 'c14')])[0,1]

                        r2_apri = r_apri**2
                        r2_apos = r_apos**2

                        ax[1,0].text(0.05, 0.95 - (i*1), f'R$^2$ apri {exp} = {r2_apri:.2f}', transform=ax[1,0].transAxes, va='top', size=12)
                        ax[1,0].text(0.05, 0.90 - (i*1), f'R$^2$ apos {exp} = {r2_apos:.2f}', transform=ax[1,0].transAxes, va='top', size=12)

                        ax[1,1].hist(db_apri.obs_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')] - db_apri.apri_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')], bins=10, label=f'{exp} Prior', density=True, alpha=0.5)
                        ax[1,1].hist(db_apri.obs_permil.loc[(db_apri.site == station) & (db_apri.tracer == 'c14')] - db_apos.apos_permil.loc[(db_apos.site == station) & (db_apos.tracer == 'c14')], bins=10, label=f'{exp} Posterior', density=True, alpha=0.5)
                        ax[1,1].set_title(f'{station} {tr} mismatch')
                        ax[1,1].set_xlim(-10, 10)
                        ax[1,1].legend()

                except:
                    pass

    f.tight_layout()

def getRegion(reg):

    Regions = {
        'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK'],
        'Northern Europe': ['DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE'],
        'Southern Europe': ['AL', 'HR', 'EL', 'IT', 'MT', 'ME', 'MK', 'PT', 'RS', 'SI', 'ES'],
        'Western Europe': ['AT', 'BE', 'FR', 'DE', 'LI', 'LU', 'NL', 'CH'],
        'Nordics': ['DK', 'FI', 'NO', 'SE', 'IS'], # Iceland is not in the domain
        '$^{14}$C Obs sites': ['DE', 'SE', 'CH', 'CZ', 'FR', 'FI'],
        'British Isles': ['UK', 'IE'],
        'Baltic States': ['EE', 'LV', 'LT', 'DK', 'SE', 'FI', 'NO'],
        'Northern Europe - IS': ['DK', 'EE', 'FI', 'IE', 'LV', 'LT', 'NO', 'SE', 'UK'],
        'Benelux': ['BE', 'NL', 'LU'],
    }

    if len(reg)>3:
        lndmsk = open_dataset('/home/carlosg/lu2023-12-19/carlos/scripts/NUTS_RG_20M_2021_3857_05deg.nc')
        idx = [where(lndmsk['country_ID'] == i)[0][0] for i in Regions[reg]]
        reg_mask = sum([lndmsk.country_fraction[i] for i in idx])
    else:
        lndmsk = open_dataset('/home/carlosg/lu2023-12-19/carlos/scripts/NUTS_RG_20M_2021_3857_05deg.nc')
        i = where(lndmsk['country_ID'] == reg)[0][0]
        reg_mask = lndmsk.country_fraction[i]

    return reg_mask

def monthlyRegion(category, alias_dict, time_span='Monthly'):

    alias = list(alias_dict.keys())

    f, ax = subplots(3, 2, figsize=(12, 8))

    ax = ax.flatten()

    Regions = ['Whole domain', '14C Obs', 'Western Europe', 'Southern Europe', 'Eastern Europe', 'Northern Europe']

    cat = category

    for i, reg in enumerate(Regions):

        if i == 0:
            reg_mask = 1
        else:
            reg_mask = getRegion(reg)

        ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span, reg_mask)
        ds_truth.emis.plot(ax=ax[i], label=f'Truth', linestyle='--', color='k', linewidth=1)

        ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/modelData.apri.nc', cat, time_span, reg_mask)

        # if i == 0:
        #     # error = read_csv(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/control.apos.csv')
        #     # error_cat = error.loc[error.category == cat.split('/')[1]].groupby('itime').sum(numeric_only=True)
        #     # error_cat.prior_uncertainty = error_cat.prior_uncertainty * 12 * 1e-18 / [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]

        #     vectors = read_csv(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/control.apos.csv')
        #     corr = File(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/matrix.h5', 'r')
        #     corr_cat = array(corr['hcor'][cat.split('/')[1]][:])
        #     err_cat = vectors.loc[vectors.category == cat.split('/')[1]]
        #     error_cat = calcTotalUncertainty(corr_cat, err_cat) * 1000 / 7 

        #     ax[i].fill_between(ds_apri.time, ds_apri.emis - error_cat, ds_apri.emis + error_cat, alpha=0.5, label=f'Prior uncertainty', color='darkred')
        # else:
        #     # ds_apri_tot, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/modelData.apri.nc', cat, time_span, 1)
        #     # error = read_csv(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/control.apos.csv')
        #     # error_cat = error.loc[error.category == cat.split('/')[1]].groupby('itime').sum(numeric_only=True)
        #     # error_cat.prior_uncertainty = error_cat.prior_uncertainty * 12 * 1e-18 * (ds_apri.emis.values / ds_apri_tot.emis.values) / [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]
        #     # ax[i].fill_between(ds_apri.time, ds_apri.emis - error_cat.prior_uncertainty.values, ds_apri.emis + error_cat.prior_uncertainty.values, alpha=0.5, label=f'Prior uncertainty', color='darkred')
        #     ax[i].fill_between(ds_apri.time, ds_apri.emis * 0.95, ds_apri.emis * 1.05, alpha=0.5, label=f'Prior ±5%', color='darkred')

        ds_apri.emis.plot(ax=ax[i], label=f'Prior', linestyle='-', color='darkred', linewidth=1)
        if 'Flat' in alias and cat == 'co2/fossil':
            al = 'Flat'
            ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[al]}/modelData.apri.nc', cat, time_span, reg_mask)
            # vectors = read_csv(f'/home/carlos/Project/LUMIA/results/{alias_dict[al]}/control.apos.csv')
            # corr = File(f'/home/carlos/Project/LUMIA/results/{alias_dict[al]}/matrix.h5', 'r')
            # corr_cat = array(corr['hcor'][cat.split('/')[1]][:])
            # err_cat = vectors.loc[vectors.category == cat.split('/')[1]]
            # error_cat = calcTotalUncertainty(corr_cat, err_cat) * 1000 / 7 

            # ax[i].fill_between(ds_apri.time, ds_apri.emis - error_cat, ds_apri.emis + error_cat, alpha=0.5, label=f'Flat prior unc', color='darkred', linestyle='dashdot')
            ds_apri.emis.plot(ax=ax[i], label=f'Flat prior', linestyle='dashdot', color='darkred', linewidth=1)

        mrkr = ['o', 'v', 's', 'D', 'X', 'P', 'd', 'p', 'h', 'H', '8', 'x']

        for j, al in enumerate(alias):
            

            ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[al]}/modelData.apos.nc', cat, time_span, reg_mask)
            ds_apos.emis.plot(ax=ax[i], label=f'{al}', linestyle='-', color='teal', marker=mrkr[j], linewidth=0.5, fillstyle='none')

        ax[i].set_title(f'{reg}')
        ax[i].set_ylabel(unit)
        ax[i].set_xlabel('')
        ax[i].grid(alpha=0.5, linestyle='--', linewidth=0.5)
        months = mdates.MonthLocator()  # set the locator to months
        date_fmt = mdates.DateFormatter('%b')  # set the formatter to month abbreviations
        ax[i].xaxis.set_major_locator(months)
        ax[i].xaxis.set_major_formatter(date_fmt)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[i].legend(loc='upper center', fontsize=8)

    f.tight_layout()


def monthlyTotal(alias_dict, time_span='Monthly'):

    alias = list(alias_dict.keys())

    f, ax = subplots(1, 1, figsize=(10, 4))

    categories = ['co2/fossil', 'co2/biosphere']
    truth_ls = ['dashed', 'dotted']
    apri_ls = ['solid', 'dashdot']

    reg_mask = 1

    for i, cat in enumerate(categories):

        print(cat)

        ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span, reg_mask)
        ds_truth.emis.plot(ax=ax, label=f'Truth {cat}', linestyle=truth_ls[i] , color='k', linewidth=1)

        ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/modelData.apri.nc', cat, time_span, reg_mask)

        error = read_csv(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias[0]]}/control.apos.csv')
        error_cat = error.loc[error.category == cat.split('/')[1]].groupby('itime').sum(numeric_only=True)
        error_cat.prior_uncertainty = error_cat.prior_uncertainty * 12 * 1e-18 / [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]
        ax.fill_between(ds_apri.time, ds_apri.emis - error_cat, ds_apri.emis + error_cat, alpha=0.5, label=f'Prior uncertainty {cat}', color='darkred', linestyle=apri_ls[i])

        ds_apri.emis.plot(ax=ax, label=f'Prior', linestyle=apri_ls[i] , color='darkred', linewidth=1)

        mrkr = ['o', 's', 'v', 'D', 'X', 'P', 'd', 'p', 'h', 'H', '8', 'x']

        for j, al in enumerate(alias):
            ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[al]}/modelData.apos.nc', cat, time_span, reg_mask)
            ds_apos.emis.plot(ax=ax, label=f'{al.split(" ")[1]} {cat}', linestyle=apri_ls[i], color='teal', marker=mrkr[j], linewidth=0.5, fillstyle='none')

    # ax.set_title(f'{reg}')
    ax.set_ylabel(unit)
    ax.set_xlabel('')
    ax.grid(alpha=0.5, linestyle='--', linewidth=0.5)
    months = mdates.MonthLocator()  # set the locator to months
    date_fmt = mdates.DateFormatter('%b')  # set the formatter to month abbreviations
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='lower right', fontsize=8)

    f.tight_layout()

def budget_byregion(category, experiments, time_span, sec_axis, regions, alias_dict):

    f, ax = subplots(2, 1, figsize=(20, 8))

    cat = category

    for i, reg in enumerate(regions):

        reg_mask = getRegion(reg)

        ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span, reg_mask)
        if i == 0:
            ds_truth.emis.plot(ax=ax[0], label=f'{reg} Truth {cat}', marker='o', linestyle='-')
        else:
            if sec_axis:
                ax2 = ax[0].twinx()
                ds_truth.emis.plot(ax=ax2, label=f'{reg} Truth {cat}', marker='o', linestyle='--')
            else:
                ds_truth.emis.plot(ax=ax[0], label=f'{reg} Truth {cat}', marker='o', linestyle='--')
        for j, alias in enumerate(experiments):
            ds_apri, unit = toTimeUnits(f'/home/carlosg/lu2023-12-19/carlos/data/output/{alias_dict[alias]}/modelData.apri.nc', cat, time_span, reg_mask)
            ds_apos, unit = toTimeUnits(f'/home/carlosg/lu2023-12-19/carlos/data/output/{alias_dict[alias]}/modelData.apos.nc', cat, time_span, reg_mask)
            if i == 0:
                ds_apri.emis.plot(ax=ax[0], label=f'{alias} Prior {cat}', marker='s')
                ds_apos.emis.plot(ax=ax[0], label=f'{alias} Posterior {cat}', marker='^')
                ax[0].legend(loc='upper left')
                ax[0].set_ylabel(unit)
            else:
                if sec_axis:
                    ds_apri.emis.plot(ax=ax2, label=f'{alias} Prior {cat}', marker='s', linestyle='--')
                    ds_apos.emis.plot(ax=ax2, label=f'{alias} Posterior {cat}', marker='^', linestyle='--')
                    ax2.legend(loc='upper right')
                    ax2.set_ylabel(unit)
                else:
                    ds_apri.emis.plot(ax=ax[0], label=f'{alias} Prior {cat}', marker='s', linestyle='--')
                    ds_apos.emis.plot(ax=ax[0], label=f'{alias} Posterior {cat}', marker='^', linestyle='--')
                    ax[0].legend(loc='upper left')
                    ax[0].set_ylabel(unit)
    
    for j, reg in enumerate(regions):

        reg_mask = getRegion(reg)

        categories = ['co2/fossil', 'co2/biosphere', 'co2/ocean']

        for i, cat in enumerate(categories):

            ds_truth, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/fwd_truth_new_preprocessed/results/modelData.fwd.nc', cat, time_span, reg_mask)

            if i == 0:
                ds_truth_sum = ds_truth
            else:
                ds_truth_sum += ds_truth

        if j == 0:
            ds_truth_sum.emis.plot(ax=ax[1], label=f'{reg} Truth', marker='o', linestyle='-')
        else:
            if sec_axis:
                ax2 = ax[1].twinx()
                ds_truth_sum.emis.plot(ax=ax2, label=f'{reg} Truth', marker='o', linestyle='--')
            else:
                ds_truth_sum.emis.plot(ax=ax[1], label=f'{reg} Truth', marker='o', linestyle='--')

        for alias in experiments:
            for i, cat in enumerate(categories):
                
                ds_apri, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apri.nc', cat, time_span, reg_mask)
                ds_apos, unit = toTimeUnits(f'/home/carlos/Project/LUMIA/results/{alias_dict[alias]}/modelData.apos.nc', cat, time_span, reg_mask)

                if i == 0:
                    ds_apri_sum = ds_apri
                    ds_apos_sum = ds_apos
                else:
                    ds_apri_sum += ds_apri
                    ds_apos_sum += ds_apos

            if j == 0:
                ds_apri_sum.emis.plot(ax=ax[1], label=f'{reg} {alias} Prior', marker='s')
                ds_apos_sum.emis.plot(ax=ax[1], label=f'{reg} {alias} Posterior', marker='v')
                ax[1].legend(loc='upper left')
                ax[1].set_ylabel(unit)
                ax[1].set_title('Total CO$_2$')
            else:
                if sec_axis:
                    ds_apri_sum.emis.plot(ax=ax2, label=f'{reg} {alias} Prior', marker='s', linestyle='--')
                    ds_apos_sum.emis.plot(ax=ax2, label=f'{reg} {alias} Posterior', marker='v', linestyle='--')
                    ax2.legend(loc='upper right')
                    ax2.set_ylabel(f'{reg} {unit}')
                else:
                    ds_apri_sum.emis.plot(ax=ax[1], label=f'{reg} {alias} Prior', marker='s', linestyle='--')
                    ds_apos_sum.emis.plot(ax=ax[1], label=f'{reg} {alias} Posterior', marker='v', linestyle='--')
                    ax[1].legend(loc='upper left')
                    ax[1].set_ylabel(unit)

        f.tight_layout()

# def annual_budget(category, experiments, time_span, sec_axis, alias_dict):




    
