import pandas as pd

from copd_forecasting.features import (
    add_temporal_features,
    build_subset_a,
    build_subset_b,
    create_binary_target,
    lag_column_name,
)


def test_create_binary_target():
    df = pd.DataFrame(
        {
            "n_urgencias": [
                0,
                1,
                2,
                0,
            ]
        }
    )

    target = create_binary_target(df)

    assert target.tolist() == [
        0,
        1,
        1,
        0,
    ]


def test_lag_column_name():
    assert (
        lag_column_name(
            "NO2",
            0,
        )
        == "NO2_LAG0"
    )

    assert (
        lag_column_name(
            "NO2",
            3,
        )
        == "NO2_LAG-3"
    )


def test_add_temporal_features():
    df = pd.DataFrame(
        {
            "fecha_asistencia": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-07-01",
                ]
            )
        }
    )

    result = add_temporal_features(df)

    assert "dia_semana" in result.columns
    assert "estac_sin" in result.columns
    assert "estac_cos" in result.columns

    assert len(result) == 2

def test_feature_subset_dimensions():
    environmental_vars = [
        "PM10",
        "PM2p5",
        "NO",
        "NO2",
        "CO",
        "O3",
        "temp_min",
        "temp_med",
        "temp_max",
        "temp_sens",
        "pres_min",
        "pres_med",
        "pres_max",
        "prec_24",
        "sol_24",
        "vel_med",
        "vel_max",
        "hume_med",
        "hume_max",
    ]

    data = {
        "fecha_asistencia": pd.date_range(
            "2023-01-01",
            periods=10,
            freq="D",
        )
    }

    for variable in environmental_vars:
        for lag in range(8):
            if lag == 0:
                column = f"{variable}_LAG0"
            else:
                column = (
                    f"{variable}_LAG-{lag}"
                )

            data[column] = [1.0] * 10

    df = pd.DataFrame(data)

    df = add_temporal_features(df)

    subset_a = build_subset_a(df)
    subset_b = build_subset_b(df)

    assert subset_a.shape[1] == 48
    assert subset_b.shape[1] == 155