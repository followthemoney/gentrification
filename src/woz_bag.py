from sqlalchemy import create_engine, text
import pandas as pd
import geopandas as gpd
import os
from dotenv import load_dotenv

load_dotenv()

PATH = os.environ.get('PATH_RAW')

def connect(db):
    '''
    Connect to PostgreSQL instance
    '''
    connection_string = f'postgresql+psycopg2://postgres:postgres@localhost/{db}'
    engine = create_engine(connection_string, pool_pre_ping=True, connect_args={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        } )
    connection = engine.connect()
    #print(f'Connected to {db}')
    
    return connection

def gdf_from_postgis(db, sql):
    '''
    Create geodataframe from PostGIS
    '''
    conn = connect(db)
    gdf = gpd.GeoDataFrame.from_postgis(text(sql), conn, geom_col='geometry')

    if not gdf.crs:
        gdf = gdf.set_crs(28992)
    else:
        gdf = gdf.to_crs(28992)

    return gdf

df = pd.read_csv(PATH + 'wozobjecten_2016_2022.csv')

ids = list(set(df.nummeraanduiding_id))

sql = f'SELECT DISTINCT \
                v.nummeraanduiding_id\
                , v.pand_id\
                , v.oppervlakte\
                , v.geom AS geometry\
        FROM vbo v\
        WHERE v.nummeraanduiding_id::numeric IN {tuple(ids)}\
        AND v.einddatum  = 0'

gdf = gdf_from_postgis('bagv2', sql).drop_duplicates()

gdf.nummeraanduiding_id = gdf.nummeraanduiding_id.astype('str')
df.nummeraanduiding_id = df.nummeraanduiding_id.apply(lambda x: str(x).zfill(16))

woz = pd.merge(gdf,
               df,
               on='nummeraanduiding_id',
               how='left')

gdf.to_file(PATH + 'bag_woz.geojson', driver='GeoJSON')
gdf.to_parquet(PATH + 'bag_woz.parquet')

woz.to_file(PATH + 'bag_wozwaarden.geojson', driver='GeoJSON')
woz.to_parquet(PATH + 'bag_wozwaarden.parquet')

