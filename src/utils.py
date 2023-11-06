from sqlalchemy import create_engine, text
import psycopg2
import geopandas as gpd

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