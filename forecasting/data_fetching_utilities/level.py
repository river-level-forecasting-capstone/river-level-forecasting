#%%
import dataretrieval.nwis as nwis
import pandas as pd

def get_historical_level(gauge_id, start="1900-01-01", end=None, parameterCd='00060', drop_cols=["00060_cd", "site_no"], rename_dict={"00060":"level"} ):
    """
    Fetch level data for the given gauge ID. Fetches instant values from start to end.
    Drops and renames columns according to given args.

    Args:
        gauge_id (string): USGS Gauge ID
        start (str, optional): Start date in the form "yyyy-mm-dd". Defaults to "1900-01-01", giving data from start of collection.
        end  (str, optional): End date in the form "yyyy-mm-dd". Defaults to None, giving data til end of collection.
        parameterCd (str, optional): Which parameter to fetch data for. Defaults to '00060' indicated mean level.
        drop_cols (list, optional): Column names to drop if they are present. Defaults to ["00060_cd", "site_no"] (useless metadata).
        rename_dict (dict, optional): Dictionary of default:new defining column renamings. Defaults to {"00060":"level"}.

    Returns:
        df (Pandas dataframe): Formatted dataframe of fetched data
    """
    # Fetch level data
    df = nwis.get_record(sites=gauge_id, service='iv', start=start, end=end, parameterCd=parameterCd)

    # Filter out any columns that are present in the drop_cols list
    drop_cols = list(filter(lambda x: x in df.columns, drop_cols))
    df.drop(columns=drop_cols, inplace=True)

    # Rename columns as specified
    df.rename(columns=rename_dict, inplace=True)
    print(df.index)
    # Convert index to datetime objects TODO: remove double conversion
    df.index = pd.to_datetime(df.index)
    print(df.index)
    df.index = df.index.map(lambda x: x.replace(tzinfo=None))
    print(df.index)
    df.index = pd.DatetimeIndex(df.index)
    print(df.index)
    # df.index = df.index.tz_convert("pst")


    # Return the formatted dataframe
    return df

df = get_historical_level("14377100", start="2019-01-01")
# %%
