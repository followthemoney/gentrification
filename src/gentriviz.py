import hvplot.pandas
import holoviews as hv
from holoviews.operation.datashader import datashade, spread
from holoviews.element import tiles
from holoviews import opts
import pandas as pd
import geopandas as gpd

def plot_maps(df16: pd.DataFrame, 
              df22: pd.DataFrame,
              min_oppervlakte,
              max_oppervlakte,
              plaats,
              income1,
              income2,
              region,
              inflation,
              overwaarde,
              mortgage_range, 
              divorced):

    opts.defaults(opts.Overlay(active_tools=['wheel_zoom']))

    params = []
    if overwaarde:
        params.append('overwaarde')
    if region:
        params.append('region')
    if inflation:
        params.append('inflation')
    if mortgage_range:
        params.append(f'inc. range {mortgage_range}')
    if divorced:
        params.append('divorced')

    title =  f'- {plaats[0]} - Waarde m2 | €{income1 + income2} | {min_oppervlakte} m2 - {max_oppervlakte} m2\n{" - ".join(params)}'
    
    # change CRS for mapping

    df16 = df16.to_crs(3857)
    df22 = df22.to_crs(3857)

    plot16 = df16.hvplot(tiles='CartoLight',
                         title=f'2016 | {len(df16)} objects {title}',
                         color='woz_2016_m2',
                         size=2,
                         cmap='jet',
                         width=600,
                         height=350,
                         clabel='woz 2016 m2')

    plot22 = df22.hvplot(tiles='CartoLight',
                         title=f'2022 | {len(df22)} objects {title}',
                         color='woz_2022_m2',
                         size=2,
                         cmap='jet',
                         width=600,
                         height=350,
                         clabel='woz 2022 m2')
   
    return (plot16 + plot22).cols(2)


    
def plot_histograms(df_2016,
                    df_2022,
                    df_ses_woa,
                    df_leefbaarometer,
                    metric):
    
    opts.defaults(opts.Overlay(active_tools=['wheel_zoom']))

    if df_2016.crs != 28992:
        df_2016 = df_2016.to_crs(28992)
    if df_2022.crs != 28992:
        df_2022 = df_2022.to_crs(28992)
    if df_ses_woa.crs != 28992:
        df_ses_woa = df_ses_woa.to_crs(28992)
    if df_leefbaarometer.crs != 28992:
        df_leefbaarometer = df_leefbaarometer.to_crs(28992)
    
    if metric in ['lbm', 'afw', 'fys', 'onv', 'soc', 'vrz', 'won']:
        df_2016 = df_2016.sjoin(df_leefbaarometer[['geometry', metric]], predicate='intersects').dropna(subset=metric)
        df_2022 = df_2022.sjoin(df_leefbaarometer[['geometry', metric]], predicate='intersects').dropna(subset=metric)

    elif metric in ['financiele_welvaart_percentielgroep',
                    'vermogen_percentielgroep', 'ses_woa_score',
                    'ses_woa_financiele_welvaart', 'ses_woa_opleidingsniveau',
                    'ses_woa_arbeidsverleden']:
        
        df_2016 = df_2016.sjoin(df_ses_woa[df_ses_woa.jaar==2016][['geometry', metric]], predicate='intersects').dropna(subset=metric)
        df_2022 = df_2022.sjoin(df_ses_woa[df_ses_woa.jaar==2021][['geometry', metric]], predicate='intersects').dropna(subset=metric)
        
    elif metric == 'woz':
        metric = 'woz_difference_postcode_5'
    
    hist16 = df_2016.hvplot.hist(y=metric,
                                 bins=30,
                                 color='red',
                                 width=600,
                                 height=350,
                                 alpha=0.5,
                                 title=f'{metric}',
                                 ).opts(show_legend=True)
    
    hist22 = df_2022.hvplot.hist(y=metric,
                                 bins=30,
                                 color='blue',
                                 width=600,
                                 height=350,
                                 alpha=0.5,
                                 title=f'{metric}').opts(show_legend=True)

    return (hist16 * hist22)