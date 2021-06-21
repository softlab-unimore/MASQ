import logging
import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.exc import SQLAlchemyError
import urllib


def check_url_connection(url_connection):
    if 'SQL Server' in url_connection:
        odbc_params = urllib.parse.quote_plus(url_connection)
        url_connection = f'mssql+pyodbc:///?odbc_connect={odbc_params}'

    return url_connection


def get_connector(url_connection):
    url_connection = check_url_connection(url_connection)
    try:
        engine = create_engine(url_connection)
        engine.connect()
        return engine
    except SQLAlchemyError as e:
        logging.error(e)
        return None


def get_tables(url_connection):
    url_connection = check_url_connection(url_connection)
    try:
        engine = create_engine(url_connection)
        meta_data = MetaData(bind=engine)
        meta_data.reflect()
        tables = meta_data.tables
        names = list(tables.keys())
        return names
    except SQLAlchemyError as e:
        logging.error(e)
        return None


def check_table(connector, table):
    return False


def check_column(url_connection, table_name, column_name):
    url_connection = check_url_connection(url_connection)
    engine = create_engine(url_connection)
    meta_data = MetaData(bind=engine)
    meta_data.reflect()
    tables = meta_data.tables

    if table_name not in tables:
        return False

    columns = [val.key for val in tables[table_name].columns]
    return column_name in columns


def get_table(url_connection, table_name):
    url_connection = check_url_connection(url_connection)
    try:
        engine = create_engine(url_connection)
        query = "select * from {}".format(table_name)
        ds = pd.read_sql(query, engine)
        return ds
    except SQLAlchemyError as e:
        logging.error(e)
        return None


def get_columns(url_connection, table_name):
    url_connection = check_url_connection(url_connection)
    try:
        # ToDo: modify query using Metadata
        engine = create_engine(url_connection)
        query = "select * from {}".format(table_name)
        ds = pd.read_sql(query, engine)
        return ds.columns.to_list()
    except SQLAlchemyError as e:
        logging.error(e)
        return None


def execute_query(url_connection, query):
    url_connection = check_url_connection(url_connection)
    engine = create_engine(url_connection)
    ds = pd.read_sql(query, engine)
    return ds


def execute_multi_queries(url_connection, queries):
    url_connection = check_url_connection(url_connection)
    engine = create_engine(url_connection)
    for q in queries[:-1]:
        for qq in q.split(';'):
            engine.execute(qq)

    ds = pd.read_sql(queries[-1], engine)

    return ds


def get_column(url_connection, table_name, column_name):
    url_connection = check_url_connection(url_connection)
    try:
        engine = create_engine(url_connection)
        with engine.connect() as connection:
            res = connection.execute("select {} from {}".format(column_name, table_name))
            labels = [x[0] for x in res]
        return labels
    except SQLAlchemyError as e:
        logging.error(e)
        return None
