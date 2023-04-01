from typing import Tuple

import streamlit as st
import pandas as pd
import subprocess
import json
from io import StringIO
import base64


def compute_fds(df, metanome_jar_path="metanome-cli-1.2-SNAPSHOT.jar"):
    temp_csv_file = "temp_input.csv"
    df.to_csv(temp_csv_file, index=False)

    algorithm_jar_path = "HyFD-1.2-SNAPSHOT.jar"
    algorithm = "de.metanome.algorithms.hyfd.HyFD"

    command = f"java -cp {metanome_jar_path}:{algorithm_jar_path} de.metanome.cli.App --algorithm {algorithm} --file-key " \
              f"INPUT_GENERATOR --files {temp_csv_file} --separator , --header -o file:test"

    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise Exception(f"Metanome execution failed with return code {process.returncode}: {stderr.decode()}")

    with open("./results/test_fds", "r") as output_file:
        lines = output_file.readlines()

    results = [json.loads(line) for line in lines]

    fds = []
    for result in results:
        determinant_columns = [col["columnIdentifier"] for col in result["determinant"]["columnIdentifiers"]]
        dependant_column = result["dependant"]["columnIdentifier"]
        fds.append((tuple(determinant_columns), dependant_column))

    return fds


def get_sub_df(df, det_cols):
    return df[list(det_cols)].drop_duplicates()


def score_fds(df, fds):
    fd_scores = {}
    df_len = len(df)
    # Iterate through the list of functional dependencies
    for fd in fds:
        # Calculate the score for the current functional dependency
        determinant, dependent = fd
        df_determinant = get_sub_df(df, determinant)

        score = df_len - len(df_determinant)

        # Add the functional dependency and its score to the dictionary
        fd_scores[fd] = score

    return fd_scores


def normalize_table(df: pd.DataFrame, selected_fd: Tuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
    determinant_columns, dependant_column = selected_fd
    # Split the original DataFrame into two DataFrames
    df_determinant = get_sub_df(df, determinant_columns)
    df_dependant = get_sub_df(df, list(determinant_columns) + [dependant_column])
    # Drop cols from the splitted DataFrame using the dependant columns
    normalized_df = df.drop(dependant_column, axis=1).drop_duplicates()
    return normalized_df, df_determinant, df_dependant


def print_df_with_stats(df):
    st.write("records: ", len(df),
             "columns: ", len(df.columns),
             "cells: ", len(df) * len(df.columns))
    st.dataframe(df)


def to_csv_download_link(df, filename="normalized_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href


st.set_page_config(page_title="FD-based Normalization", layout="wide")
st.title("Functional Dependency-based Normalization Tool")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    file_string = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(file_string)

    st.subheader("Original Data")
    print_df_with_stats(df)

    fds = compute_fds(df)
    fd_scores = score_fds(df, fds)

    str_scores = {str(fd): score for fd, score in fd_scores.items()}

    # Convert the scores dictionary to a pandas DataFrame
    scores_df = pd.DataFrame(list(str_scores.items()), columns=['Functional Dependency', 'Score'])

    # Set the index to Functional Dependency
    scores_df.set_index('Functional Dependency', inplace=True)
    scores_df = scores_df.sort_values('Score', ascending=False)
    # Streamlit app
    st.title('Functional Dependencies Score Visualization')
    st.write('The scores for each functional dependency are shown in the bar chart below:')

    col1, col2 = st.columns([1, 3])
    with col1:
        st.dataframe(scores_df)

    with col2:
        st.bar_chart(scores_df)

    fd_score_tuples = sorted([(k, v) for k, v in fd_scores.items()], key=lambda x: x[1], reverse=True)

    selected_fd = st.selectbox("Select a Functional Dependency for normalization:", fd_score_tuples,
                               format_func=lambda x: f"{x[0]} (score: {x[1]})")

    if st.button("Normalize"):
        normalized_df, df_determinant, df_dependant = normalize_table(df, selected_fd[0])
        st.subheader("Normalized Data")

        col1, col2 = st.columns(2)

        # Display the DataFrames in their respective columns
        with col1:
            print_df_with_stats(normalized_df)

        with col2:
            print_df_with_stats(df_dependant)
