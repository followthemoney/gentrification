import warnings; warnings.filterwarnings('ignore')
import json
from dotenv import load_dotenv
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm
import os
import pandas as pd
import geopandas as gpd

from utils import connect, gdf_from_postgis

load_dotenv()

PATH = os.environ.get('PATH_RAW')

def get_woz(path_in: str,
            path_out: str) -> pd.DataFrame:
   
    '''Imports and parses WOZ-values 
    from a very large JSON file
    
    PARAMETERS
    ----------
    path_in: path to JSON file
    path_out: path to parquet file with WOZ values
    '''
    
    with open(path_in, 'r') as file:
        wozlist = []
        for line in tqdm(file):
            wozdict = {}
            woz = json.loads(line)
            wozwaarden = woz.get('wozWaarden')
            wozobjecten = woz.get('wozObject')
            if wozobjecten is not None:
                nummeraanduiding_id = wozobjecten.get('nummeraanduidingid')
                if nummeraanduiding_id is not None:
                    wozdict.update({'nummeraanduiding_id': int(nummeraanduiding_id),
                                    'postcode': wozobjecten.get('postcode')})
                else:
                    continue
            else:
                continue

            # Get values for 2016 and 2022. If there is no value, take 0 (house is probably not built yet, or demolished)
            for waarde in wozwaarden:
                if waarde.get('peildatum') == '2016-01-01':
                    wozdict.update({'woz_2016': waarde.get('vastgesteldeWaarde')})
                else:
                    wozdict.update({'woz_2016': 0})
                if waarde.get('peildatum') == '2022-01-01': 
                    wozdict.update({'woz_2022': waarde.get('vastgesteldeWaarde')})
                else:
                    wozdict.update({'woz_2022': 0})
            # Remove empty or invalid values
            if wozdict.get('woz_2022') and wozdict.get('woz_2016') is not None:
                wozlist.append(wozdict)

    df = pd.DataFrame(wozlist)
    del wozlist
    # Write to parquet
    df.to_parquet(path_out)

    return df


def get_bag(path_in: str,
            path_out: str) -> gpd.GeoDataFrame:
    
    '''Connect to PostGIS and get BAG
    objects using nummeraanduiding-id from WOZ-data
    
    PARAMETERS
    ----------
    path_in: path to parsed WOZ data
    path_out: path where parquet file needs to be stored
    '''
    
    # Import WOZ data
    df = pd.read_parquet(path_in)

    # Get all unique nummeraanduiding ids
    ids = list(set(df.nummeraanduiding_id))

    # Create query
    sql = f'SELECT DISTINCT \
                    v.nummeraanduiding_id\
                    , v.pand_id\
                    , v.oppervlakte\
                    , v.geom AS geometry\
            FROM vbo v\
            WHERE v.nummeraanduiding_id::numeric IN {tuple(ids)}\
            AND v.einddatum  = 0 \
            AND v.oppervlakte <= 250 \
            AND v.statuscode IN (3, 4, 7)' # Filter data to make set smaller

    # Get data
    gdf = gdf_from_postgis('bagv2', sql).drop_duplicates()

    #Write to parquet
    gdf.to_parquet(path_out)

    return gdf
    
def merge_bag_woz(path_left: str,
                  path_right: str) -> gpd.GeoDataFrame:
    
    '''Merges WOZ objects with BAG objects on 
    nummeraanduiding id
    
    PARAMETERS
    ----------
    path_left: path to dataframe WOZ data (non geometric)
    path_right: path to geodataframe BAG objects
    '''
    
    if os.path.isfile(path_left) and os.path.isfile(path_right):
        print('starting woz-bag merge...')
        
        df = pd.read_parquet(path_right)
        gdf = gpd.read_parquet(path_left)
        print('...imported woz and bag')
        
        # Clean nummeraanduiding id for merging
        df.nummeraanduiding_id = df.nummeraanduiding_id.apply(lambda x: str(x).zfill(16)) #Nummeraanduiding_id needs padded zeroes
        gdf.nummeraanduiding_id = gdf.nummeraanduiding_id.astype('str')

        gdf = pd.merge(gdf,
                        df,
                        on='nummeraanduiding_id',
                        how='left')
        print('...merged woz and bag')
        
        # We need some metrics for later
        gdf['woz_difference_rel'] = round(gdf['woz_2022'] / gdf['woz_2016'], 2)

        # Drop houses with extreme WOZ differences (often due to houses getting a low WOZ value during construction)
        gdf = gdf[gdf.woz_difference_rel <= 4]

        # Get more metrics
        gdf['woz_2016_m2'] = round(gdf['woz_2016'] / gdf['oppervlakte'])
        gdf['woz_2022_m2'] = round(gdf['woz_2022'] / gdf['oppervlakte'])
        gdf['woz_difference_abs'] = gdf['woz_2022'] - gdf['woz_2016']

        # Get the median rise in WOZ values in postcode 5 area (postcode 4 is a rather large area, postcode 6 is quite small)
        gdf['woz_difference_postcode_5'] = gdf.groupby(gdf.postcode.str[0:5]).woz_difference_rel.transform('median')
        print('...calculated relevant metrics')

        # Drop houses with extreme WOZ differences (often due to houses getting a low WOZ value during construction)
        gdf = gdf[gdf.woz_difference_rel <= 4]

        # Add gemeente-naam
        if os.path.isfile(PATH + 'processed/gemeenten.parquet'):
            gemeenten = gpd.read_parquet(PATH + 'processed/gemeenten.parquet')
            gdf = gdf.sjoin(gemeenten[['gemeente', 'geometry']], predicate='intersects', how='left')
            gdf.drop('index_right', axis=1, inplace=True)
        else:
            raise ValueError('Cannot join with gemeenten. First import CBS gemeenten.')
        print('...added gemeenten to woz-bag')

        # Write file to parquet
        gdf.to_parquet(PATH + 'processed/bag_wozwaarden.parquet')
    else:
        raise ValueError(f'WOZ or BAG data not yet collected. Please download these first')
    
    print(f'...finished importing {len(gdf)} rows!')
    return gdf


def get_cbs(path_out: str) -> gpd.GeoDataFrame:

    '''Gets all geometries of buurten between 2016 and 2023
    
    PARAMETERS
    ----------
    path_out: path to store the geodataframe in parquet format
    '''
    
    # Neighborhood codes and geometries change over time so we need to collect everything
    cbs_2016 = gpd.read_file(PATH + 'cbs/wijkbuurt2016/buurt_2016.shp')
    cbs_2016['jaar'] = 2016
    cbs_2016 = cbs_2016[cbs_2016.WATER=='NEE'].copy()

    cbs_2017 = gpd.read_file(PATH + 'cbs/wijkbuurt2017/buurt_2017_v3.shp')
    cbs_2017['jaar'] = 2017
    cbs_2017 = cbs_2017[cbs_2017.WATER=='NEE'].copy()

    cbs_2018 = gpd.read_file(PATH + 'cbs/wijkbuurt2018/buurt_2018_v3.shp')
    cbs_2018['jaar'] = 2018
    cbs_2018 = cbs_2018[cbs_2018.H2O=='NEE'].copy()

    cbs_2019 = gpd.read_file(PATH + 'cbs/wijkbuurt2019/buurt_2019_v3.shp')
    cbs_2019['jaar'] = 2019
    cbs_2019= cbs_2019[cbs_2019.H2O=='NEE'].copy()

    cbs_2020 = gpd.read_file(PATH + 'cbs/wijkbuurt2020/buurt_2020_v3.shp')
    cbs_2020['jaar'] = 2020
    cbs_2020 = cbs_2020[cbs_2020.H2O=='NEE'].copy()

    cbs_2021 = gpd.read_file(PATH + 'cbs/wijkbuurt2021/buurt_2021_v2.shp')
    cbs_2021['jaar'] = 2021
    cbs_2021 = cbs_2021[cbs_2021.H2O=='NEE'].copy()

    cbs_2022 = gpd.read_file(PATH + 'cbs/wijkbuurt2022/buurt_2022_v1.shp')
    cbs_2022['jaar'] = 2022
    cbs_2022 = cbs_2022[cbs_2022.H2O=='NEE'].copy()

    cbs_2023 = gpd.read_file(PATH + 'cbs/wijkbuurt2023/buurt_2023_v0.shp')
    cbs_2023['jaar'] = 2023
    cbs_2023 = cbs_2023[cbs_2023.WATER=='NEE'].copy()

    # Voeg alles samen en verwijder dubbelingen
    buurten = [cbs_2016, cbs_2017, cbs_2018, cbs_2019, cbs_2020,cbs_2021, cbs_2022, cbs_2023]

    gdf = pd.concat(buurten)
    gdf = gdf[['BU_CODE', 'BU_NAAM', 'jaar', 'geometry']].copy()
    gdf = gdf.rename(columns={'BU_CODE': 'buurtcode',
                            'BU_NAAM': 'buurtnaam'})
    gdf = gdf.sort_values(by='jaar', ascending=False)
    gdf = gdf.drop_duplicates(subset=['buurtcode'], keep='last').reset_index(drop=True) #Drop duplicates and keep the most recent one
    gdf.drop('jaar', axis=1, inplace=True)

    gdf.to_parquet(path_out)

    # No reason to keep all the dataframes around
    for buurt in buurten:
        del buurt

    return gdf


def get_ses_woa(path_in: str,
                path_out: str,
                cbs_path: str) -> gpd.GeoDataFrame:
    
    '''Get SES-WOA data provided by CBS and 
    combine them with CBS buurt information and
    geometries.

    PARAMETERS
    ----------
    path_in: path to ses-woa data
    path_out: path to geodataframe in parquet format
    cbs_path: path to cbs geometries
    '''
    
    # Import SES-WOA dat
    ignorecols = ['Gestandaardiseerd inkomen/Gemiddelde percentielgroep (Getal)',
                  'Spreiding/Spreiding totaal/Waarde (Getal)',
                  'Spreiding/Spreiding welvaart/Waarde (Getal)',
                  'Spreiding/Spreiding opleidingsniveau/Waarde (Getal)',
                  'Spreiding/Spreiding arbeidsverleden/Waarde (Getal)']

    df = pd.read_csv(path_in, sep=';')
    df.drop(ignorecols, axis=1, inplace=True)

    # Clean columns
    for col in df.columns[3:]:
        df[col] = df[col].str.replace(',', '.').astype('float')

    cols = {'Perioden': 'jaar',
            'Wijken en buurten': 'naam',
            'Regiocode (gemeente, wijk, buurt) (code)': 'regiocode',
            'Financiële Welvaart/Gemiddelde percentielgroep (Getal)': 'financiele_welvaart_percentielgroep',
            'Vermogen/Gemiddelde percentielgroep (Getal)': 'vermogen_percentielgroep',
            'SES-WOA/Totaalscore/Gemiddelde score (Getal)': 'ses_woa_score',
            'SES-WOA/Deelscore financiële welvaart/Gemiddelde score (Getal)': 'ses_woa_financiele_welvaart',
            'SES-WOA/Deelscore opleidingsniveau/Gemiddelde score (Getal)': 'ses_woa_opleidingsniveau',
            'SES-WOA/Deelscore arbeidsverleden/Gemiddelde score (Getal)': 'ses_woa_arbeidsverleden'}

    df = df.rename(columns=cols)

    # Select buurten from 2019 (to fit data to)
    df = df[(df['regiocode'].str.contains('BU')) & ((df.jaar==2016) | (df.jaar==2021))].copy()

    # Import CBS
    if os.path.isfile(cbs_path):
        cbs = gpd.read_parquet(cbs_path)
    else:
        raise ValueError('First import CBS data')

    # Create geodataframe
    gdf = pd.merge(df,
                   cbs,
                   left_on='regiocode',
                   right_on='buurtcode',
                   how='left')
    
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=28992)
    
    # Write to parquet
    gdf.to_parquet(path_out)

    return gdf

def get_leefbaarometer(path_left: str,
                       path_right: str,
                       path_out: str) -> gpd.GeoDataFrame:
    
    '''Get metrics of leefbaarometer, which needs to be 
    made spatial with the provided geofile
    
    PARAMETERS
    ----------
    path_left: path to leefbaarometer metrics
    path_right: path to leefbaarometer geometries
    path_out: path to where to store the resulting geodataframe
    '''
    
    # Read metrics
    df = pd.read_csv(path_left)

    # Read geometries
    gdf = gpd.read_file(path_right, crs=28992)

    # Merge
    df = pd.merge(df[df.jaar == 2020].copy(),
                  gdf[['bu_code', 'geometry']],
                  on='bu_code',
                  how='right')
    
    # Create geodataframe
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=28992)
    gdf.drop(['versie', 'jaar'], axis=1, inplace=True)
    gdf = gdf.rename(columns={'bu_code': 'regiocode'})

    # Write to parquet
    gdf.to_parquet(path_out)

    return gdf

def get_overwaarde(path_in: str,
                   path_out: str) -> pd.DataFrame:
    
    '''Import overwaarde file
    
    PARAMETERS
    ----------
    path_in: path to CBS data
    path_out: path to stored dataframe
    '''
    
    # Read file
    df = pd.read_csv(path_in, sep=';', decimal=',')

    # Calculate overwaarde
    df['overwaarde'] = df.gem_woningwaarde - df.gem_hypotheekschuld

    # Rename columns and normalize overwaarde
    cols = ['gem_hypotheekschuld', 'gem_woningwaarde', 'bruto_inkomen', 'overwaarde']
    df[cols] = df[cols] * 1000

    # Write to file
    df.to_parquet(path_out)

    return df

def get_gemeenten(path_in: str,
                  path_out: str) -> gpd.GeoDataFrame:
    
    '''
    Get all gemeenten geometries:
    
    PARAMETERS
    ----------
    path_in: path to geometries
    path_out: path to stored geodatafrae
    '''
    
    # Get file
    if os.path.isfile(path_in):
        gdf = gpd.read_file(path_in, layer='gemeente_gegeneraliseerd')
        gdf = gdf[['statnaam', 'geometry']].copy()
        gdf = gdf.rename(columns={'statnaam': 'gemeente'})
        gdf.to_parquet(path_out)
    else:
        raise ValueError('Gemeenten geopackage is missing in CBS folder')
    
    return gdf

def import_data(data: str,
                existing: bool = True):
    
    '''
    Imports data. If dataset already exists,
    the stored data is used
    '''
    
    # Check dataset
    accepted_data_values = ['woz', 'bag', 'woz_bag', 'cbs', 'ses_woa', 'leefbaarometer', 'overwaarde', 'gemeenten']
    
    if data not in accepted_data_values:
        raise ValueError(f'Param "data" should be in {", ".join(accepted_data_values)}')
    else:
        pass
    
    # If existing, point to right dataset
    if existing:
        if data == 'woz':
            df = pd.read_parquet(PATH +'processed/wozobjecten_2016_2022.parquet')
        elif data == 'bag':
            df = gpd.read_parquet(PATH + 'processed/woz_bag.parquet')
        elif data == 'woz_bag': 
            df = gpd.read_parquet(PATH + 'processed/bag_wozwaarden.parquet')
        elif data == 'cbs': 
            df = gpd.read_parquet(PATH + 'processed/cbs.parquet')
        elif data == 'ses_woa': 
            df = gpd.read_parquet(PATH + 'processed/ses_woa.parquet')
        elif data == 'leefbaarometer':
            df = gpd.read_parquet(PATH + 'processed/leefbaarometer.parquet')
        elif data == 'overwaarde':
            df = pd.read_parquet(PATH + 'processed/overwaarde.parquet')
        elif data == 'gemeenten':
            df = gpd.read_parquet(PATH + 'processed/gemeenten.parquet')

    # If not existing, create the datasets and store them     
    else:
        if data == 'woz':
            df = get_woz(path_in=PATH + 'woz.json', 
                         path_out=PATH + 'processed/wozobjecten_2016_2022.parquet')
        elif data == 'bag':
            df = get_bag(path_in=PATH + 'processed/wozobjecten_2016_2022.parquet',
                         path_out=PATH + 'processed/woz_bag.parquet')
        elif data == 'woz_bag':
        
            df = merge_bag_woz(path_right=PATH + 'processed/wozobjecten_2016_2022.parquet',
                               path_left=PATH + 'processed/woz_bag.parquet')       
        elif data == 'cbs':
            df = get_cbs(PATH + 'processed/cbs.parquet')
        
        elif data == 'ses_woa':
            df = get_ses_woa(path_in=PATH + 'cbs/ses.csv',
                             path_out=PATH + 'processed/ses_woa.parquet',
                             cbs_path=PATH + 'processed/cbs.parquet')
        elif data == 'leefbaarometer':
            df = get_leefbaarometer(path_left=PATH + 'leefbaarometer/Leefbaarometer 3.0 - meting 2020 - scores buurt.csv',
                                    path_right=PATH + 'leefbaarometer/geometrie-leefbaarometer-3-0/buurt 2020.shp',
                                    path_out=PATH + 'processed/leefbaarometer.parquet')
        elif data == 'overwaarde':
            df = get_overwaarde(path_in=PATH + 'cbs/overwaarde.csv',
                                path_out=PATH + 'processed/overwaarde.parquet')
        elif data == 'gemeenten':
            df = get_gemeenten(path_in=PATH + 'cbs/cbsgebiedsindelingen2023.gpkg',
                                  path_out=PATH + 'processed/gemeenten.parquet')

    return df

        
def get_max_mortgage(income1: int, income2: Optional[int] = 0, divorced=False):

    '''
    Calculates max mortgage based on NIBUD
    models

    PARAMETERS
    ----------
    income1: income partner 1
    income2: income partner 2
    divorced: True if partners are divorced
    '''

    # First define some constants
    ANNUITY_2016 = 0.00424 #Based on NIBUD
    ANNUITY_2022 = 0.00389

    SALARY_2016 = 0.8367 #Also based on NIBUD
    SALARY_2022 = 1

    # Calculate financieringslastpercentage
    flp_2016 = pd.read_csv(PATH + 'nibud/flp_2016.csv', sep=';')
    flp_2022 = pd.read_csv(PATH + 'nibud/flp_2022.csv', sep=';')

    # Clean it up
    flp_2016.bruto_inkomen = flp_2016.bruto_inkomen.str.replace('-', '0').str.replace('.', '').astype('int')
    flp_2016.flp_2016 = flp_2016.flp_2016.str.replace(',', '.').astype('float')
    flp_2022.flp_2022 = flp_2022.flp_2022.str.replace(',', '.').astype('float')

    # Sort incomes from low to high
    if income2 > 0:
        incomes = sorted([income1, income2])
        income_2016 = incomes[1] + (incomes[0] * 0.5) # In 2016, the lowest income wasn't fully used in calculating max mortgage
        income_2022 = incomes[1] + incomes[0]

        if divorced:
            income_2016 = incomes[0]
            income_2022 = incomes[0]
    
    else:
        income_2016 = income1 
        income_2022 = income1
    
    # Normalise salary
    income_2016 = income_2016 * SALARY_2016
    income_2022 = income_2022 * SALARY_2022

    # Find financieringslastpercentage for bruto inkomen
    _flp_2016 = flp_2016[(flp_2016.bruto_inkomen < income_2016 +1)].iloc[-1].flp_2016
    _flp_2022 = flp_2022[(flp_2022.bruto_inkomen < income_2022 +1)].iloc[-1].flp_2022

    # Find the max mortgage
    max_mortgage_2016 = round(income_2016 * (_flp_2016 / 12 / ANNUITY_2016))
    max_mortgage_2022 = round(income_2022 * (_flp_2022 / 12 / ANNUITY_2022))

    return max_mortgage_2016, max_mortgage_2022


def get_regio(plaats: List) -> List:

    '''
    Finds adjacent gemeenten to chosen region

    PARAMETERS
    ----------
    plaats: list of gemeenten
    '''

    # Import gemeenten
    gdf = gpd.read_parquet(PATH + 'processed/gemeenten.parquet')
    gdf['neighbor'] = None
    gdf = gdf[['gemeente', 'geometry', 'neighbor']].copy()

    # Find adjacent plaatsen
    regio = []
    for p in plaats:
        i = gdf[gdf.gemeente == p].index[0]
        geom = gdf.iloc[i].geometry
        regio.extend(gdf[gdf.intersects(geom)].gemeente.tolist())

    return regio            


def get_overwaarde(income, year):
    '''
    Gets overwaarde for a certain income
    based on CBS percentiles

    PARAMETERS
    ----------
    income: the full income of both partners
    year: 2016 or 2022.
    '''

    df = pd.read_parquet(PATH + 'processed/overwaarde.parquet')
    idx = df[df.jaar==year].overwaarde.sub(income).abs().idxmin()
    return round(df.iloc[idx].overwaarde)
    
            
def compare_jaren(gdf: gpd.GeoDataFrame,
                  min_oppervlakte: Optional[int],
                  max_oppervlakte: Optional[int],
                  plaats: List[str],
                  income1: int,
                  income2: Optional[int] = 0,
                  region: bool = False,
                  inflation: bool = True,
                  mortgage_range=0.9,
                  overwaarde=True,
                  divorced=False,
                  ) -> Tuple[Dict, Dict]:
    
    '''
    Outputs two filtered dataframes, one of 2016 and one of 2022
    based on the params provided. The dataframes are wrapped in a 
    dictionary with some calculated values (overwaarde, max_mortgage)
    for validation purposes.
    
    Parameters
    ----------
    gdf: a geodataframe of WOZ values
    min_oppervlakte: minimal area of house (m2)
    max_oppervlakte: maximal area of house (m2)
    plaats: list of gemeenten. If all gemeenten are needed than provide ['all']
    income1: income first partner (used for max_mortgage calculation based on NIBUD)
    income2: income second partner (used for max_mortgage calculation basedon NIBUD)
    region: if True all adjecent gemeenten are filtered out as well
    inflation: if True WOZ-values are adjusted for inflation
    mortgage_range: a range of values used for searching for house as a percentage
    overwaarde: if True overwaarde is taken into account (source CBS)
    divorced: if True lowest income is taken and overwaarde is halved
    '''
    
    # Get oppervlakte
    if min_oppervlakte == None:
        min_oppervlakte = 0
    if max_oppervlakte == None:
        max_oppervlakte == 250 # Set a max oppervlakte to exclude outliers
    
    # Get plaats and region    
    if region:
        plaatsen = get_regio(plaats)
    else:
        plaatsen = plaats
        
    if 'all' in plaats:
        plaatsen = list(set(gdf.gemeente))
    else:
        if not isinstance(plaats, list):
            raise ValueError('Plaats parameter should be "all" or a list of gemeenten') 
        
    # Get max mortgage
    max_2016, max_2022 = get_max_mortgage(income1, income2, divorced)
    
    # Calculate inflation
    if inflation:
        max_2016 *= 1
        max_2022 *= 0.86
        
    # Add overwaarde to mortgage if applicable
    if divorced and overwaarde:
        overwaarde_2016 = get_overwaarde(income1 + income2, 2016) / 2
        overwaarde_2022 = get_overwaarde(income1 + income2, 2022) / 2
        max_2016 += overwaarde_2016
        max_2022 += overwaarde_2022

    if overwaarde and not divorced:
        overwaarde_2016 = get_overwaarde(income1 + income2, 2016)
        overwaarde_2022 = get_overwaarde(income1 + income2, 2022)
        max_2016 += overwaarde_2016
        max_2022 += overwaarde_2022

    else:
        overwaarde_2016 = 0
        overwaarde_2022 = 0

    
    # Filter WOZ values    
    df_2016 = gdf[(gdf[f'woz_2016'] >= max_2016*mortgage_range) & (gdf[f'woz_2016'] <= max_2016) & (gdf.gemeente.isin(plaatsen)) & (gdf.oppervlakte >= min_oppervlakte) & (gdf.oppervlakte <= max_oppervlakte)].copy()
    df_2022 = gdf[(gdf[f'woz_2022'] >= max_2022*mortgage_range) & (gdf[f'woz_2022'] <= max_2022) & (gdf.gemeente.isin(plaatsen)) & (gdf.oppervlakte >= min_oppervlakte) & (gdf.oppervlakte <= max_oppervlakte)].copy()

    # Create dictionaries
    dict_2016 = {'df': df_2016,
                 'max_mortgage': max_2016,
                 'overwaarde': overwaarde_2016}
    dict_2022 = {'df': df_2022,
                 'max_mortgage': max_2022,
                 'overwaarde': overwaarde_2022}
    
    return dict_2016, dict_2022

