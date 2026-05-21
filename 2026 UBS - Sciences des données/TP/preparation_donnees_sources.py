from datetime import timedelta, datetime
from zipfile import ZipFile
from io import BytesIO
import requests

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    concat,
    read_csv,
    to_datetime,
    date_range,
)

# Electricity from RTE for Bretagne region #####################################
ELEC_DATA_SOURCES = [
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Bretagne_Annuel-Definitif_2023.zip",
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Bretagne_Annuel-Definitif_2024.zip",
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Bretagne_En-cours-Consolide.zip",
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Bretagne_En-cours-TR.zip",
]


def download_and_extract_zip(url):
    # Download the zip file
    response = requests.get(url)
    if response.status_code == 200:
        # Read zip and parse
        with ZipFile(BytesIO(response.content)) as zip_file:
            elec_table = read_csv(
                zip_file.open(zip_file.filelist[0].filename),
                sep="\t",
                encoding="latin-1",
                usecols=["Date", "Heures", "Consommation"],
            )
        # Rename columns for consistency
        elec_table.rename(
            columns={
                "Date": "date",
                "Heures": "heure",
                "Consommation": "consommation_elec",
            },
            inplace=True,
        )
        # Combine date and time into a single datetime column
        elec_table["datetime"] = to_datetime(
            elec_table["date"] + " " + elec_table["heure"],
            format="%Y-%m-%d %H:%M",
            errors="raise",
        )
        # Suppose it is Paris time
        elec_table["datetime"] = elec_table["datetime"].dt.tz_localize(
            "Europe/Paris", nonexistent="shift_forward", ambiguous="NaT"
        )

        # Drop rows if time ends with 15 and 45 ==> They are all missing
        elec_table = elec_table[~elec_table["heure"].str.endswith((":15", ":45"))]

        # Clean the table: drop useless columnes, set datetime as index and row the last row (missing value rows)
        elec_table.drop(columns=["date", "heure"], inplace=True)
        elec_table.set_index("datetime", inplace=True)
        elec_table = elec_table[elec_table.index.notna()]

        # Apply types
        elec_table["consommation_elec"] = elec_table["consommation_elec"].astype(float)

        return elec_table
    else:
        raise RuntimeError(
            f"Failed to download file from {url}. Status code: {response.status_code}"
        )


def collecte_electricite():
    # Download and extract all the zip files, concatenate them, sort by datetime and drop rows with missing values in the consumption column
    electricity_data = (
        concat(map(download_and_extract_zip, ELEC_DATA_SOURCES), axis=0)
        .sort_index()
        .dropna(subset=["consommation_elec"])
    )

    return electricity_data


# Météo prévisionnelle pour la Bretagne à J+2 ##################################
def collecte_meteo():
    # Get from open-meteo
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 48.0684,
        "longitude": -2.8,
        "hourly": "temperature_2m_previous_day2",
        # Collect as UTC, to next convert to paris time
        "timezone": "UTC",
        "start_date": "2022-11-10",
        "end_date": (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d"),
    }
    response = requests.get(url, params=params)

    # Prepare dataframe with the relevant data
    # Read the response as JSON and extract the relevant data
    weather_forecast = DataFrame(
        response.json()["hourly"],
        columns=["time", "temperature_2m_previous_day2"],
    ).rename(
        columns={"time": "datetime", "temperature_2m_previous_day2": "temperature"}
    )

    # parse datetime
    weather_forecast.index = to_datetime(
        weather_forecast["datetime"],
        format="%Y-%m-%dT%H:%M",
        utc=True,
    ).dt.tz_convert("Europe/Paris")

    weather_forecast.drop(columns=["datetime"], inplace=True)

    # Type of temperature column
    weather_forecast["temperature"] = weather_forecast["temperature"].astype(float)

    return weather_forecast


# Vacances scolaires en France pour la zone B ##################################
def parse_holiday_datetimes(series: Series) -> Series:
    return to_datetime(
        series,
        format="ISO8601",
        utc=True,
    ).dt.tz_convert("Europe/Paris")


def collecte_vacances():
    # Parse file with pandas
    url = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire@dataeducation/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
    vacations = read_csv(
        url,
        sep=";",
        usecols=[
            "Description",
            "Date de début",
            "Date de fin",
            "Zones",
        ],
    )
    vacations.rename(
        columns={
            "Description": "description",
            "Date de début": "start_date",
            "Date de fin": "end_date",
            "Zones": "zone",
        },
        inplace=True,
    )
    # Encode datatime as date
    vacations["start_date"] = parse_holiday_datetimes(vacations["start_date"]).dt.floor(
        "D"
    )
    vacations["end_date"] = parse_holiday_datetimes(vacations["end_date"]).dt.floor("D")
    # remove one day because datetime go to the hours of the next day
    vacations["end_date"] = vacations["end_date"] - timedelta(days=1)

    # Create range for each period from date de debut to date de fin, and explode the dataframe to have one row per day
    vacations["date"] = vacations.apply(
        lambda row: date_range(start=row["start_date"], end=row["end_date"], freq="D"),
        axis=1,
    )

    vacations = vacations.explode("date")
    # Drop start_date and end_date columns, and tidy
    vacations.drop(columns=["start_date", "end_date"], inplace=True)
    vacations.dropna(subset=["date"], inplace=True)
    # keep uique rows, sort by date and set date as index
    vacations = vacations.drop_duplicates()

    # vacations.sort_values("date", inplace=True)
    vacations.set_index("date", inplace=True)

    return vacations


def collecte_et_prepare_donnees_tp():
    # electricite
    electricite = collecte_electricite()
    # meteo
    meteo = collecte_meteo()
    # vacances
    vacances = collecte_vacances()
    # Keep only zone B
    vacances = vacances[vacances["zone"] == "Zone B"]

    # join them
    # insert meteo
    integrated_data = (
        # join into electricity and keep outer to keep meteo beyond the last timestamp of electricity data
        electricite.join(meteo, how="outer")
        # add date for integrating holiday
        .assign(
            date=lambda d: to_datetime(d.index).floor("D"),
        )
        # insert holiday on the this date
        .join(vacances, how="left", on="date")
        # If description of vacances is not null, it means it is a holiday
        .assign(vacances=lambda d: d["description"].notna())
        .drop(columns=["date", "zone", "description"])
    )

    # Drop rows for half hours in the datetime (no temperature data for these rows)
    integrated_data = integrated_data[integrated_data.index.minute == 0]
    # Sort on index
    integrated_data.sort_index(inplace=True)
    # Control if regular grid from first to last index value, with a frequency of 1 hour
    expected_index = date_range(
        start=integrated_data.index.min(), end=integrated_data.index.max(), freq="h"
    )
    # Identify missing timestamps in the integrated dataset and fill them with NaN values if less than 10 missing timestamps, otherwise raise an error
    missing_timestamps = expected_index.difference(integrated_data.index)

    if (not missing_timestamps.empty) & (len(missing_timestamps) < 10):
        # Create missing rows while preserving original dtypes
        missing_data = integrated_data.iloc[:0].reindex(missing_timestamps)
        # Append the missing data to the integrated dataset and sort by index
        integrated_data = concat([integrated_data, missing_data], axis=0).sort_index()
        integrated_data["vacances"] = integrated_data["vacances"].astype(bool)
    elif len(missing_timestamps) >= 10:
        raise RuntimeError(
            f"Too many missing timestamps in the integrated dataset: {len(missing_timestamps)}. Missing timestamps: {missing_timestamps}"
        )
    else:
        pass
    # Drop duplicates if any, and control if there are duplicated timestamps in the integrated dataset
    integrated_data = integrated_data[~integrated_data.index.duplicated(keep="first")]

    # sort and carry forward the temperature and vacances values for the missing timestamps (if any)
    integrated_data["consommation_elec"] = integrated_data["consommation_elec"].ffill(
        limit_area="inside"
    )  # Do not fill beyond the last available data point, to avoid filling with old data for future timestamps
    integrated_data["vacances"] = integrated_data["vacances"].ffill()

    # Cut data prior the first available data of electricity: one need to have the same time range for all train features
    integrated_data = integrated_data[integrated_data.index >= electricite.index.min()]

    # Cut data after last available electricity consumption data + 48 h
    integrated_data = integrated_data[
        integrated_data.index <= (electricite.index.max() + timedelta(hours=48))
    ]

    # insert holiday
    return integrated_data


def ajouter_indicateurs_temporels(df: DataFrame) -> DataFrame:
    datetime_index = (
        df.index if isinstance(df.index, DatetimeIndex) else to_datetime(df.index)
    )

    # Add second in day
    df["minute_in_day"] = datetime_index.hour * 60 + datetime_index.minute
    # Add day in week
    df["day_of_week"] = datetime_index.dayofweek
    # Day in year
    df["day_of_year"] = datetime_index.dayofyear

    return df


if __name__ == "__main__":
    # Backup dataset in parquet
    collecte_et_prepare_donnees_tp().to_parquet("offline_data/integrated_data.parquet")
