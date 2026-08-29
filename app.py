import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="WPBL Expansion Predictor",
    layout="wide",
)

st.title("WPBL Expansion Location Predictor")
st.write(
    "Rank potential WPBL expansion markets using stadium capacity, women's "
    "baseball history, international-airport access, and population."
)
st.caption(
    "This is a transparent opportunity index, not a statistically calibrated "
    "probability of expansion."
)


# ============================================================
# DATA CLEANING
# ============================================================

def clean_number(value):
    """Convert values such as '5,000', '5,000 seats', or 5000 to float."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace("$", "")
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)

    if not match:
        return np.nan

    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return np.nan


def clean_yes_no(value):
    """Convert common Yes/No representations to 1/0."""
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()

    return int(
        text in {
            "yes",
            "y",
            "true",
            "1",
            "1.0",
            "x",
            "available",
        }
    )


def normalize_column_name(name):
    """Make column matching tolerant of capitalization and punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def prepare_dataframe(data):
    """
    Convert the supplied CSV into the standard columns used by the model.

    The supplied WPBL dataset uses:
    source_id, stadium, capacity, city, state, current_team, league,
    population, womens_team, international_airport
    """

    df = data.copy()

    # Normalize aliases by a punctuation-insensitive key.
    aliases = {
        "sourceid": "source_id",
        "stadium": "stadium",
        "stadium": "stadium",
        "venue": "stadium",
        "capacity": "capacity",
        "city": "city",
        "state": "state",
        "currentteam": "current_team",
        "team": "current_team",
        "league": "league",
        "pop": "population",
        "population": "population",
        "citypopulation": "population",
        "womens team": "womens_team",
        "womensteam": "womens_team",
        "baseball": "womens_team",
        "womens team history": "womens_team",
        "womens_team": "womens_team",
        "internationalairport": "international_airport",
        "internationalairportaccess": "international_airport",
        "airport": "international_airport",
        "international_airport": "international_airport",
    }

    rename = {}

    for column in df.columns:
        key = normalize_column_name(column)

        # Handle curly apostrophe / punctuation variants through
        # normalized keys.
        if key in aliases:
            rename[column] = aliases[key]

    df = df.rename(columns=rename)

    required = [
        "stadium",
        "city",
        "state",
        "capacity",
        "population",
        "womens_team",
        "international_airport",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
            + ".\n\n"
            + "Columns found: "
            + ", ".join(map(str, data.columns))
        )

    # Clean numeric fields.
    df["capacity"] = df["capacity"].apply(clean_number)
    df["population"] = df["population"].apply(clean_number)

    # Clean binary fields.
    df["womens_team"] = df["womens_team"].apply(clean_yes_no)
    df["international_airport"] = df["international_airport"].apply(
        clean_yes_no
    )

    # Text fields.
    for column in ["stadium", "city", "state"]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Do NOT silently discard the whole dataset.
    # Only retain rows with the core market identifiers and numeric values.
    before = len(df)

    df = df[
        (df["stadium"] != "")
        & (df["city"] != "")
        & (df["state"] != "")
        & df["capacity"].notna()
        & df["population"].notna()
    ].copy()

    df = df.reset_index(drop=True)

    return df, before


# ============================================================
# SCORING MODEL
# ============================================================

def capacity_fit(capacity, target):
    """
    1.0 when capacity equals target.
    0.0 when capacity is 2,500+ seats away.
    """

    if pd.isna(capacity):
        return 0.0

    difference = abs(float(capacity) - float(target))

    return max(0.0, 1.0 - difference / 2500.0)


def population_fit(population):
    """
    Logarithmic population score with diminishing returns.
    1.0 corresponds to approximately 1 million people or more.
    """

    if pd.isna(population) or population <= 0:
        return 0.0

    return min(
        1.0,
        np.log1p(float(population)) / np.log1p(1_000_000),
    )


def calculate_scores(
    df,
    target_capacity,
    capacity_weight,
    womens_weight,
    airport_weight,
    population_weight,
):
    result = df.copy()

    total_weight = (
        capacity_weight
        + womens_weight
        + airport_weight
        + population_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "At least one feature weight must be greater than zero."
        )

    result["capacity_fit"] = result["capacity"].apply(
        lambda value: capacity_fit(value, target_capacity)
    )

    result["population_fit"] = result["population"].apply(
        population_fit
    )

    result["capacity_points"] = (
        result["capacity_fit"] * capacity_weight
    )

    result["womens_points"] = (
        result["womens_team"] * womens_weight
    )

    result["airport_points"] = (
        result["international_airport"] * airport_weight
    )

    result["population_points"] = (
        result["population_fit"] * population_weight
    )

    result["raw_points"] = (
        result["capacity_points"]
        + result["womens_points"]
        + result["airport_points"]
        + result["population_points"]
    )

    result["prediction_score"] = (
        result["raw_points"] / total_weight * 100
    ).round(1)

    result["capacity_gap"] = (
        result["capacity"] - target_capacity
    ).abs()

    result["market"] = (
        result["city"] + ", " + result["state"]
    )

    result["likelihood"] = pd.cut(
        result["prediction_score"],
        bins=[-0.01, 49.99, 64.99, 79.99, 100.01],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
    )

    # Rank after sorting so ranks are guaranteed to correspond to
    # the displayed order.
    result = result.sort_values(
        ["prediction_score", "capacity_gap"],
        ascending=[False, True],
    ).reset_index(drop=True)

    result["rank"] = np.arange(1, len(result) + 1)

    return result


def estimated_likelihood(score, sensitivity):
    """
    Presentation-only transformation of the opportunity score.
    This is NOT a calibrated probability.
    """

    z = ((float(score) - 60.0) / 9.0) * sensitivity

    return 100.0 / (1.0 + np.exp(-z))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Prediction Controls")

upload = st.sidebar.file_uploader(
    "Upload a compatible CSV",
    type=["csv"],
)

target_capacity = st.sidebar.slider(
    "Target stadium capacity",
    min_value=2500,
    max_value=12000,
    value=5000,
    step=100,
)

st.sidebar.subheader("Feature Weights")

capacity_weight = st.sidebar.slider(
    "Stadium capacity",
    min_value=0,
    max_value=100,
    value=30,
)

womens_weight = st.sidebar.slider(
    "Women's team history",
    min_value=0,
    max_value=100,
    value=25,
)

airport_weight = st.sidebar.slider(
    "International airport",
    min_value=0,
    max_value=100,
    value=20,
)

population_weight = st.sidebar.slider(
    "Population",
    min_value=0,
    max_value=100,
    value=25,
)

sensitivity = st.sidebar.slider(
    "Likelihood sensitivity",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

st.sidebar.caption(
    f"Total model weight: "
    f"{capacity_weight + womens_weight + airport_weight + population_weight}"
)


# ============================================================
# LOAD DATA
# ============================================================

try:
    if upload is not None:
        raw = pd.read_csv(upload)
        source_description = "Uploaded CSV"
    else:
        data_path = Path(__file__).with_name(
            "wpbl_stadium_data.csv"
        )

        if not data_path.exists():
            st.error(
                "Could not find wpbl_stadium_data.csv next to app.py."
            )
            st.info(
                "Place the CSV in the same folder as app.py, "
                "or upload it using the sidebar."
            )
            st.stop()

        raw = pd.read_csv(data_path)
        source_description = "wpbl_stadium_data.csv"

except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()


if raw.empty:
    st.error(
        f"{source_description} was successfully opened, "
        "but it contains zero rows."
    )
    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

try:
    df, rows_before_cleaning = prepare_dataframe(raw)
except Exception as error:
    st.error(f"Could not prepare the dataset: {error}")

    with st.expander("Show detected CSV columns"):
        st.write(list(raw.columns))

    st.stop()


# This is the critical protection against the
# "single positional indexer is out-of-bounds" error.
if df.empty:
    st.error(
        "No usable markets remain after cleaning the CSV."
    )

    st.write(
        f"Rows in CSV before cleaning: {rows_before_cleaning}"
    )
    st.write(
        f"Rows remaining after cleaning: {len(df)}"
    )

    st.write("Columns detected:")
    st.write(list(raw.columns))

    st.warning(
        "The model requires valid values for stadium, city, state, "
        "capacity, and population."
    )

    st.stop()


# ============================================================
# CALCULATE SCORES
# ============================================================

try:
    scored = calculate_scores(
        df,
        target_capacity,
        capacity_weight,
        womens_weight,
        airport_weight,
        population_weight,
    )

except Exception as error:
    st.error(
        f"Could not calculate market predictions: {error}"
    )
    st.stop()


# Second protection against an empty DataFrame.
if scored.empty:
    st.error(
        "The scoring model returned zero markets."
    )
    st.stop()


scored["estimated_likelihood"] = scored[
    "prediction_score"
].apply(
    lambda value: estimated_likelihood(
        value,
        sensitivity,
    )
)


# ============================================================
# TOP MARKET
# ============================================================

# At this point scored has been explicitly checked.
top = scored.iloc[0]


# ============================================================
# KPI DASHBOARD
# ============================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Markets Analyzed",
    len(scored),
)

k2.metric(
    "Top Market",
    top["market"],
)

k3.metric(
    "Top Score",
    f"{top['prediction_score']:.1f}/100",
)

k4.metric(
    "Estimated Likelihood",
    f"{top['estimated_likelihood']:.0f}%",
)


# ============================================================
# TABS
# ============================================================

dashboard, simulator, comparison, details = st.tabs(
    [
        "Prediction Dashboard",
        "Market Simulator",
        "Market Comparison",
        "Model Details",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with dashboard:

    st.subheader("Predicted WPBL Expansion Markets")

    max_markets = min(30, len(scored))

    if max_markets >= 5:
        top_n = st.slider(
            "Markets shown",
            min_value=5,
            max_value=max_markets,
            value=min(15, max_markets),
        )
    else:
        top_n = len(scored)
        st.caption(
            f"Showing all {top_n} available markets."
        )

    chart_df = scored.head(top_n).copy()

    fig = px.bar(
        chart_df.sort_values("prediction_score"),
        x="prediction_score",
        y="market",
        orientation="h",
        text="prediction_score",
        hover_data=[
            "capacity",
            "population",
            "capacity_gap",
            "womens_team",
            "international_airport",
        ],
        labels={
            "prediction_score": "Prediction Score",
            "market": "Market",
            "capacity": "Stadium Capacity",
            "population": "Population",
            "capacity_gap": "Seats From Target",
        },
        title="WPBL Expansion Prediction Score",
    )

    fig.update_xaxes(range=[0, 100])

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Market Ranking")

    ranking = scored[
        [
            "rank",
            "market",
            "stadium",
            "capacity",
            "population",
            "womens_team",
            "international_airport",
            "capacity_points",
            "womens_points",
            "airport_points",
            "population_points",
            "prediction_score",
            "estimated_likelihood",
            "likelihood",
        ]
    ].copy()

    ranking = ranking.rename(
        columns={
            "rank": "Rank",
            "market": "Market",
            "arestadiumna": "Stadium",
            "capacity": "Capacity",
            "population": "Population",
            "womens_team": "Women's Team",
            "international_airport": "International Airport",
            "capacity_points": "Capacity Points",
            "womens_points": "Women's Team Points",
            "airport_points": "Airport Points",
            "population_points": "Population Points",
            "prediction_score": "Score",
            "estimated_likelihood": "Estimated Likelihood",
            "likelihood": "Potential",
        }
    )

    ranking["Capacity"] = (
        ranking["Capacity"]
        .round()
        .astype(int)
        .map(lambda x: f"{x:,}")
    )

    ranking["Population"] = (
        ranking["Population"]
        .round()
        .astype(int)
        .map(lambda x: f"{x:,}")
    )

    ranking["Women's Team"] = ranking[
        "Women's Team"
    ].map(
        {
            1: "Yes",
            0: "No",
        }
    )

    ranking["International Airport"] = ranking[
        "International Airport"
    ].map(
        {
            1: "Yes",
            0: "No",
        }
    )

    ranking["Score"] = ranking["Score"].map(
        lambda x: f"{x:.1f}"
    )

    ranking["Estimated Likelihood"] = ranking[
        "Estimated Likelihood"
    ].map(
        lambda x: f"{x:.0f}%"
    )

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Prediction Results",
        data=scored.to_csv(index=False).encode(
            "utf-8"
        ),
        file_name="wpbl_expansion_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# SIMULATOR
# ============================================================

with simulator:

    st.subheader(
        "Hypothetical Market Simulator"
    )

    st.write(
        "Create a hypothetical WPBL market and immediately "
        "see how its score changes."
    )

    left, right = st.columns(2)

    with left:

        hypothetical_city = st.text_input(
            "City",
            "New Market",
        )

        hypothetical_state = st.text_input(
            "State",
            "XX",
        )

        hypothetical_capacity = st.number_input(
            "Stadium Capacity",
            min_value=1000,
            max_value=30000,
            value=5000,
            step=100,
        )

        hypothetical_population = st.number_input(
            "City Population",
            min_value=1000,
            max_value=10000000,
            value=500000,
            step=10000,
        )

    with right:

        hypothetical_womens = st.checkbox(
            "Has had a women's baseball team",
            value=True,
        )

        hypothetical_airport = st.checkbox(
            "Has international airport access",
            value=True,
        )

    hypothetical = pd.DataFrame(
        [
            {
                "stadium": "Hypothetical Stadium",
                "city": hypothetical_city,
                "state": hypothetical_state,
                "capacity": hypothetical_capacity,
                "population": hypothetical_population,
                "womens_team": int(
                    hypothetical_womens
                ),
                "international_airport": int(
                    hypothetical_airport
                ),
            }
        ]
    )

    simulated = calculate_scores(
        hypothetical,
        target_capacity,
        capacity_weight,
        womens_weight,
        airport_weight,
        population_weight,
    ).iloc[0]

    simulated_likelihood = estimated_likelihood(
        simulated["prediction_score"],
        sensitivity,
    )

    st.divider()

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Prediction Score",
        f"{simulated['prediction_score']:.1f}/100",
    )

    h2.metric(
        "Estimated Likelihood",
        f"{simulated_likelihood:.0f}%",
    )

    h3.metric(
        "Capacity Gap",
        f"{simulated['capacity_gap']:,.0f} seats",
    )

    components = pd.DataFrame(
        {
            "Factor": [
                "Stadium Capacity",
                "Women's Team History",
                "International Airport",
                "Population",
            ],
            "Points": [
                simulated["capacity_points"],
                simulated["womens_points"],
                simulated["airport_points"],
                simulated["population_points"],
            ],
        }
    )

    fig_components = px.bar(
        components,
        x="Factor",
        y="Points",
        text="Points",
        title="Hypothetical Market Score Components",
    )

    fig_components.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig_components,
        use_container_width=True,
    )


# ============================================================
# COMPARISON
# ============================================================

with comparison:

    st.subheader(
        "🗺️ Interactive Market Comparison"
    )

    selected_markets = st.multiselect(
        "Select markets to compare",
        scored["market"].tolist(),
        default=scored["market"]
        .head(min(5, len(scored)))
        .tolist(),
    )

    if not selected_markets:

        st.info(
            "Select one or more markets above."
        )

    else:

        comparison_df = scored[
            scored["market"].isin(selected_markets)
        ].copy()

        # Explicitly retain population because it is used
        # as the Plotly bubble size.
        comparison_chart = comparison_df[
            [
                "market",
                "city",
                "state",
                "stadium",
                "capacity",
                "population",
                "capacity_gap",
                "prediction_score",
                "likelihood",
            ]
        ].copy()

        fig_comparison = px.scatter(
            comparison_chart,
            x="capacity",
            y="prediction_score",
            size="population",
            size_max=50,
            color="likelihood",
            hover_name="market",
            hover_data={
                "capacity": ":,",
                "population": ":,",
                "capacity_gap": ":,",
                "prediction_score": ":.1f",
            },
            labels={
                "capacity": "Stadium Capacity",
                "population": "Population",
                "prediction_score": "Prediction Score",
                "likelihood": "Potential",
            },
            title=(
                "Stadium Capacity vs. WPBL Expansion Score"
            ),
        )

        fig_comparison.add_vline(
            x=target_capacity,
            line_dash="dash",
            annotation_text=(
                f"Target = {target_capacity:,}"
            ),
        )

        fig_comparison.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            fig_comparison,
            use_container_width=True,
        )

        st.subheader(
            "Selected Market Components"
        )

        comparison_table = comparison_df[
            [
                "market",
                "capacity",
                "population",
                "capacity_points",
                "womens_points",
                "airport_points",
                "population_points",
                "prediction_score",
                "estimated_likelihood",
            ]
        ].rename(
            columns={
                "market": "Market",
                "capacity": "Capacity",
                "population": "Population",
                "capacity_points": "Capacity Points",
                "womens_points": "Women's Team Points",
                "airport_points": "Airport Points",
                "population_points": "Population Points",
                "prediction_score": "Prediction Score",
                "estimated_likelihood": "Estimated Likelihood",
            }
        )

        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL DETAILS
# ============================================================

with details:

    st.subheader(
        "🧠 How the Prediction Model Works"
    )

    st.markdown(
        f"""
### 1. Stadium capacity — {capacity_weight} points

The model compares each stadium to the selected target of
**{target_capacity:,} seats**.

The closer the stadium is to the target, the higher the capacity score.

### 2. Women's baseball history — {womens_weight} points

Markets with a recorded women's baseball team receive the full
category weight.

### 3. International airport — {airport_weight} points

Markets with international-airport access receive the full category
weight.

### 4. Population — {population_weight} points

Population receives a continuous score using a logarithmic
transformation, which creates diminishing returns at very large
population levels.

### Final score

The four weighted components are normalized to a **0–100
Expansion Prediction Score**.

**80–100:** Very High

**65–79.9:** High

**50–64.9:** Moderate

**0–49.9:** Low

### Important limitation

The "Estimated Likelihood" shown in the app is a mathematical
transformation of the opportunity score. It is **not a calibrated
statistical probability**.

A true predictive probability model would require historical
expansion outcomes or other labeled outcomes.
"""
    )

    weights = pd.DataFrame(
        {
            "Factor": [
                "Stadium Capacity",
                "Women's Team History",
                "International Airport",
                "Population",
            ],
            "Weight": [
                capacity_weight,
                womens_weight,
                airport_weight,
                population_weight,
            ],
        }
    )

    fig_weights = px.bar(
        weights,
        x="Factor",
        y="Weight",
        text="Weight",
        title="Current Model Weights",
    )

    fig_weights.update_traces(
        texttemplate="%{text}",
        textposition="outside",
    )

    st.plotly_chart(
        fig_weights,
        use_container_width=True,
    )


# ============================================================
# FOOTER / DIAGNOSTICS
# ============================================================

st.sidebar.divider()

st.sidebar.success(
    f"Dataset loaded: {len(df)} usable markets"
)

with st.sidebar.expander("Dataset diagnostics"):

    st.write(
        f"Raw rows: {len(raw)}"
    )

    st.write(
        f"Usable rows: {len(df)}"
    )

    st.write(
        f"Rows removed: "
        f"{len(raw) - len(df)}"
    )

    st.write(
        "Columns:"
    )

    st.write(
        list(raw.columns)
    )
