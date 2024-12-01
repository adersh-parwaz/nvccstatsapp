import streamlit as st
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

# Set up logging for error tracking
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

def main():
    st.set_page_config(page_title="NVCC Player Stats", layout="wide")
    
    # Apply custom CSS styles
    apply_custom_styles()
    
    st.markdown("<h1 style='text-align: center; color: #800000; font-size:64px;'>NVCC Player Stats</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'excel_file' not in st.session_state:
        st.session_state.excel_file = None
    if 'excel_sheets' not in st.session_state:
        st.session_state.excel_sheets = []
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = None
    if 'player_list' not in st.session_state:
        st.session_state.player_list = []
    if 'selected_stats' not in st.session_state:
        st.session_state.selected_stats = []
    if 'year_var' not in st.session_state:
        st.session_state.year_var = 'ALL TIME'

    # Default stats (updated)
    default_stats = [
        "Matches", "Innings", "Runs", "High", "Not Out", "Avg",
        "50s", "100s", "Ducks", "Catches", "Wickets", "C+RO+S",
        "Avg.1", "Econ", "SR", "Best", "Full Overs"
    ]

    # Mapping of actual stat names to display names (updated)
    stat_display_names = {
        'Avg': 'Batting Avg',
        'Avg.1': 'Bowling Avg',
        'Econ': 'Bowling Econ',
        'SR': 'Bowling SR',
        'C+RO+S': 'Catches+RunOuts+Stumpings'
    }

    # Stats that need to be formatted to two decimal places
    two_decimal_stats = {'Avg', 'Avg.1', 'Econ', 'SR'}

    # Add color options
    highlight_colors = {
        "Light Green": "#90EE90",
        "Light Blue": "#ADD8E6",
        "Light Yellow": "#FFFFE0",
        "Light Pink": "#FFB6C1",
        "Light Orange": "#FFD580"
    }

    bar_colors = {
        "Blue": "#3498db",
        "Green": "#2ecc71",
        "Red": "#e74c3c",
        "Purple": "#9b59b6",
        "Orange": "#e67e22",
        "Gray": "#95a5a6"
    }

    title_colors = {
        "Black": "#000000",
        "Blue": "#0000FF",
        "Red": "#FF0000",
        "Green": "#008000",
        "Purple": "#800080",
        "Orange": "#FFA500"
    }

    st.sidebar.header("Navigation")

    # Load the Excel file directly from the file system
    excel_file_path = 'cricket_stats.xlsx'  # Replace with your actual file name
    if not st.session_state.excel_file:
        try:
            st.session_state.excel_file = pd.ExcelFile(excel_file_path)
            st.session_state.excel_sheets = st.session_state.excel_file.sheet_names
            # Exclude specific sheets if needed
            relevant_sheets = [sheet for sheet in st.session_state.excel_sheets if sheet not in ["Career Achievements"]]
            st.session_state.excel_sheets = relevant_sheets  # Update the session state with relevant sheets
            # Load the default sheet
            default_sheet = "ALL TIME" if "ALL TIME" in relevant_sheets else relevant_sheets[0]
            st.session_state.year_var = default_sheet
            st.sidebar.success("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load file '{excel_file_path}': {e}")
            st.sidebar.error(f"Failed to load file '{excel_file_path}': {e}")
            return

    if st.session_state.excel_file:
        # Year selection
        year_options = [sheet for sheet in st.session_state.excel_sheets if sheet != "Career Achievements"]
        st.sidebar.subheader("Select Year")
        selected_year = st.sidebar.selectbox("Select Sheet (Year):", year_options, index=year_options.index(st.session_state.year_var))
        if selected_year != st.session_state.year_var:
            st.session_state.year_var = selected_year
            st.session_state.stats_data = None

        # Load data from the selected sheet
        if st.session_state.stats_data is None:
            try:
                df = st.session_state.excel_file.parse(sheet_name=st.session_state.year_var)
                df.columns = df.columns.str.strip()
                st.session_state.stats_data = df
                st.session_state.player_list = df['Player'].dropna().tolist()
                # Apply default stats if no stats have been selected yet
                if not st.session_state.selected_stats:
                    st.session_state.selected_stats = [stat for stat in default_stats if stat in df.columns]
                else:
                    # Ensure selected_stats are valid for the current data
                    st.session_state.selected_stats = [stat for stat in st.session_state.selected_stats if stat in df.columns]
            except Exception as e:
                logging.error(f"Failed to load sheet '{st.session_state.year_var}': {e}")
                st.error(f"Failed to load sheet '{st.session_state.year_var}': {e}")

        # Navigation
        app_mode = st.sidebar.radio("Choose the app mode", ["Player Comparison", "Custom Report Generator", "Player Stats Over Years", "Top Players Over Years"])

        if app_mode == "Player Comparison":
            player_comparison(st.session_state)
        elif app_mode == "Custom Report Generator":
            custom_report_generator(st.session_state)
        elif app_mode == "Player Stats Over Years":
            player_stats_over_years(st.session_state)
        elif app_mode == "Top Players Over Years":
            top_players_over_years(st.session_state)
    else:
        st.write(f"Please ensure the data file '{excel_file_path}' exists in the app directory.")

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Main content area */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #e0e0e0; 
        }
        ::-webkit-scrollbar-thumb {
            background: #888; 
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555; 
        }
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-size: 1.5em !important;
            color: #333333;
        }
        /* Buttons */
        .stButton>button {
            color: white;
            background-color: #333333;
            border-radius: 5px;
            font-size: 16px;
        }
        /* Success messages */
        .st-bf {
            background-color: #d4edda !important;
            border-color: #c3e6cb !important;
            color: #155724 !important;
        }
        /* Increase font size of input labels */
        label {
            font-size: 1.2em;
            color: #333333;
        }
        /* Table font size */
        .dataframe th, .dataframe td {
            font-size: 1.1em;
            color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)

def player_comparison(session_state):
    st.markdown(f"<h2 style='color: #2c3e50; font-size:32px;'>{session_state.year_var} Career Analysis</h2>", unsafe_allow_html=True)

    # Mapping of actual stat names to display names (updated)
    stat_display_names = {
        'Avg': 'Batting Avg',
        'Avg.1': 'Bowling Avg',
        'Econ': 'Bowling Econ',
        'SR': 'Bowling SR',
        'C+RO+S': 'Catches+RunOuts+Stumpings'
    }

    two_decimal_stats = {'Avg', 'Avg.1', 'Econ', 'SR'}

    # Select Stats to Display
    with st.sidebar.expander("Select Stats to Display", expanded=False):
        available_stats = [col for col in session_state.stats_data.columns if col != 'Player']
        # Update available_stats to include only stats present in the DataFrame
        available_stats = [stat for stat in available_stats if stat in session_state.stats_data.columns]
        selected_stats = st.multiselect("Select Stats:", [stat_display_names.get(stat, stat) for stat in available_stats],
                                        default=[stat_display_names.get(stat, stat) for stat in session_state.selected_stats])
        # Map display names back to actual stat names
        reverse_stat_display_names = {v: k for k, v in stat_display_names.items()}
        session_state.selected_stats = [reverse_stat_display_names.get(stat, stat) for stat in selected_stats]

    # Player Selection
    st.sidebar.subheader("Player Selection")
    search_text1 = st.sidebar.text_input("Search Player 1:")
    player_options1 = [player for player in session_state.player_list if search_text1.lower() in player.lower()]
    selected_player1 = st.sidebar.selectbox("Select Player 1:", player_options1)

    compare_player2 = st.sidebar.checkbox("Compare with second player")
    if compare_player2:
        search_text2 = st.sidebar.text_input("Search Player 2:")
        player_options2 = [player for player in session_state.player_list if search_text2.lower() in player.lower()]
        selected_player2 = st.sidebar.selectbox("Select Player 2:", player_options2)
    else:
        selected_player2 = None

    compare_player3 = st.sidebar.checkbox("Compare with third player")
    if compare_player3:
        search_text3 = st.sidebar.text_input("Search Player 3:")
        player_options3 = [player for player in session_state.player_list if search_text3.lower() in player.lower()]
        selected_player3 = st.sidebar.selectbox("Select Player 3:", player_options3)
    else:
        selected_player3 = None

    # Check for duplicate players
    selected_players = [selected_player1]
    if selected_player2:
        selected_players.append(selected_player2)
    if selected_player3:
        selected_players.append(selected_player3)

    if len(selected_players) != len(set(selected_players)):
        st.error("Please select different players for comparison.")
        return

    # Display Stats
    if session_state.stats_data is not None:
        stats1, stats2, stats3 = None, None, None
        if selected_player1:
            player_data1 = session_state.stats_data[session_state.stats_data['Player'] == selected_player1]
            stats1 = player_data1.iloc[0] if not player_data1.empty else None
        if selected_player2:
            player_data2 = session_state.stats_data[session_state.stats_data['Player'] == selected_player2]
            stats2 = player_data2.iloc[0] if not player_data2.empty else None
        if selected_player3:
            player_data3 = session_state.stats_data[session_state.stats_data['Player'] == selected_player3]
            stats3 = player_data3.iloc[0] if not player_data3.empty else None

        display_stats(stats1, stats2, stats3, session_state.selected_stats, stat_display_names, two_decimal_stats)

def display_stats(stats1, stats2, stats3, selected_stats, stat_display_names, two_decimal_stats):
    cols = [col for col in [stats1, stats2, stats3] if col is not None]
    num_players = len(cols)
    if num_players == 0:
        st.write("Please select at least one player to display stats.")
        return

    # Gather player names and stats
    headers = []
    stats_list = []
    if stats1 is not None:
        headers.append(stats1['Player'])
        stats_list.append(stats1)
    if stats2 is not None:
        headers.append(stats2['Player'])
        stats_list.append(stats2)
    if stats3 is not None:
        headers.append(stats3['Player'])
        stats_list.append(stats3)

    # Build the data
    data = {}
    for stat_name in selected_stats:
        display_stat_name = stat_display_names.get(stat_name, stat_name)
        values = []
        for stats in stats_list:
            if stats is None:
                continue
            if stat_name == 'Best':
                value = get_best_stat(stats)
            else:
                value = stats.get(stat_name, "N/A")
                value = "N/A" if pd.isna(value) else value
                if stat_name in two_decimal_stats:
                    value = format_value(value)
            values.append(value)
        data[display_stat_name] = values

    df_display = pd.DataFrame(data, index=headers).transpose()

    # Apply styling
    st.table(df_display.style.set_properties(**{'text-align': 'center', 'font-size': '1.2em'}).set_table_styles(
        [{'selector': 'th', 'props': [('font-weight', 'bold'), ('font-size', '1.2em')]}]
    ))

def format_value(value):
    if value != "N/A":
        try:
            value = float(value)
            if pd.isna(value):
                return "N/A"
            return f"{value:.2f}"
        except (ValueError, TypeError):
            return "N/A"
    else:
        return value

def get_best_stat(stats):
    if stats is not None and 'Best' in stats and not pd.isna(stats['Best']):
        best_stat = str(stats['Best'])
        # Include additional 'Best' columns if they exist
        for col in ['Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27']:
            if col in stats and not pd.isna(stats[col]):
                best_stat += f" {stats[col]}"
        return best_stat
    else:
        return "N/A"

def custom_report_generator(session_state):
    st.markdown("<h2 style='color: #2c3e50; font-size:32px;'>Custom Report Generator</h2>", unsafe_allow_html=True)

    stat_display_names = {
        'Avg': 'Batting Avg',
        'Avg.1': 'Bowling Avg',
        'Econ': 'Bowling Econ',
        'SR': 'Bowling SR',
        'C+RO+S': 'Catches+RunOuts+Stumpings'
    }

    two_decimal_stats = {'Avg', 'Avg.1', 'Econ', 'SR'}

    # Report Heading
    report_heading = st.text_input("Enter Report Heading:")

    # Stat selection
    available_stats = [col for col in session_state.stats_data.columns if col != 'Player']
    display_stat_options = [stat_display_names.get(stat, stat) for stat in available_stats]
    display_stat = st.selectbox("Select Stat to Sort By:", display_stat_options)
    # Map display name back to actual stat name
    reverse_stat_display_names = {v: k for k, v in stat_display_names.items()}
    stat = reverse_stat_display_names.get(display_stat, display_stat)

    # Sort order selection
    sort_order = st.radio("Select Sort Order:", ["Most to Least", "Least to Most"])

    # Additional stats selection
    selected_additional_stats = st.multiselect("Select Additional Stats to Display:", display_stat_options)
    selected_additional_stats = [reverse_stat_display_names.get(s, s) for s in selected_additional_stats if s != display_stat]

    # Highlight color selection
    highlight_color_name = st.selectbox("Select Highlight Color:", list(highlight_colors.keys()))
    highlight_color = highlight_colors[highlight_color_name]

    generate_button = st.button("Generate Report")
    if generate_button:
        generate_custom_report(session_state.stats_data, stat, sort_order, selected_additional_stats, two_decimal_stats,
                               report_heading, highlight_color, stat_display_names)

def generate_custom_report(stats_data, stat, sort_order, selected_additional_stats, two_decimal_stats,
                           report_heading, highlight_color, stat_display_names):
    if stat not in stats_data.columns:
        st.warning(f"The selected stat '{stat}' is not available.")
        return

    # Ensure the main stat is included
    columns_to_display = ['Player', stat] + selected_additional_stats
    columns_to_display = list(dict.fromkeys(columns_to_display))  # Remove duplicates while preserving order
    report_data = stats_data[columns_to_display].copy()

    # Convert stat columns to numeric where possible
    for col in [stat] + selected_additional_stats:
        report_data[col] = pd.to_numeric(report_data[col], errors='coerce')

    # Drop players with missing stat values
    report_data.dropna(subset=[stat], inplace=True)

    # Sort the DataFrame
    ascending = True if sort_order == "Least to Most" else False
    report_data.sort_values(by=stat, ascending=ascending, inplace=True)

    # Reset index to get ranking
    report_data.reset_index(drop=True, inplace=True)

    # Display the report
    if report_heading:
        st.markdown(f"<h3 style='color: #2c3e50; font-size:28px;'>{report_heading}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #2c3e50; font-size:28px;'>Custom Report</h3>", unsafe_allow_html=True)

    # Display the report in tabular format with ranking
    report_data.insert(0, 'Rank', report_data.index + 1)
    display_columns = ['Rank', 'Player'] + columns_to_display[1:]
    display_columns_names = ['Rank', 'Player'] + [stat_display_names.get(col, col) for col in columns_to_display[1:]]

    report_data.columns = display_columns_names

    # Highlight the main stat column
    def highlight_col(x):
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        df[stat_display_names.get(stat, stat)] = f'background-color: {highlight_color}'
        return df

    st.dataframe(report_data.style.apply(highlight_col, axis=None)
                 .format({stat_display_names.get(s, s): '{:.2f}' for s in two_decimal_stats if stat_display_names.get(s, s) in report_data.columns})
                 .set_properties(**{'text-align': 'center', 'font-size': '1.1em'})
                 .set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('font-size', '1.2em')]}]))

def player_stats_over_years(session_state):
    st.markdown("<h2 style='color: #2c3e50; font-size:32px;'>Player Stats Over Years</h2>", unsafe_allow_html=True)

    # Get unique players across all sheets
    all_players = set()
    for sheet in session_state.excel_sheets:
        if sheet != "ALL TIME":
            df = session_state.excel_file.parse(sheet_name=sheet)
            # Convert to strings and remove any leading/trailing whitespace
            players = df['Player'].astype(str).str.strip()
            # Remove any empty or NaN values
            players = players[players.notna() & (players != '')]
            all_players.update(players)

    # Convert to list and sort (after ensuring all items are strings)
    all_players = sorted(list(all_players))

    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        selected_player1 = st.selectbox("Select Player:", all_players)
    with col2:
        compare_player2 = st.checkbox("Compare with another player")
        if compare_player2:
            selected_player2 = st.selectbox("Select Player 2:", all_players)
        else:
            selected_player2 = None

    # Stat selection
    default_stats = [
        "Matches", "Innings", "Runs", "High", "Not Out", "Avg",
        "50s", "100s", "Ducks", "Catches", "Wickets", "C+RO+S",
        "Avg.1", "Econ", "SR", "Best", "Full Overs"
    ]
    stat_display_names = {
        'Avg': 'Batting Avg',
        'Avg.1': 'Bowling Avg',
        'Econ': 'Bowling Econ',
        'SR': 'Bowling SR',
        'C+RO+S': 'Catches+RunOuts+Stumpings'
    }
    stat_options = [stat for stat in default_stats if stat != 'Player']
    display_stat_options = [stat_display_names.get(stat, stat) for stat in stat_options]
    display_stat = st.selectbox("Select Stat:", display_stat_options)
    reverse_stat_display_names = {v: k for k, v in stat_display_names.items()}
    stat = reverse_stat_display_names.get(display_stat, display_stat)

    # Additional stats selection
    selected_additional_stats = st.multiselect("Select Additional Stats to Display:", display_stat_options)
    selected_additional_stats = [reverse_stat_display_names.get(s, s) for s in selected_additional_stats]

    # Chart title
    chart_heading = st.text_input("Enter Chart Title:")

    # Bar color selection
    bar_color_name = st.selectbox("Select Bar Color:", list(bar_colors.keys()))
    bar_color = bar_colors[bar_color_name]

    generate_button = st.button("Generate Chart")
    if generate_button:
        generate_player_stats_chart(session_state, selected_player1, selected_player2, stat, selected_additional_stats,
                                    chart_heading, bar_color, stat_display_names)

def generate_player_stats_chart(session_state, player1, player2, stat, selected_additional_stats,
                                chart_heading, bar_color, stat_display_names):
    two_decimal_stats = {'Avg', 'Avg.1', 'Econ', 'SR'}

    # Collect data across years for player 1
    years = []
    stat_values1 = []
    additional_stats_values1 = []
    for sheet in session_state.excel_sheets:
        if sheet != "ALL TIME":
            df = session_state.excel_file.parse(sheet_name=sheet)
            df.columns = df.columns.str.strip()
            if 'Player' in df.columns and stat in df.columns:
                player_data = df[df['Player'] == player1]
                if not player_data.empty:
                    value = player_data.iloc[0][stat]
                    if pd.isna(value):
                        continue  # Skip NaN values
                    try:
                        value = float(value)
                        if stat in two_decimal_stats:
                            value = float(f"{value:.2f}")
                    except (ValueError, TypeError):
                        continue  # Skip invalid values
                    years.append(sheet)
                    stat_values1.append(value)
                    # Collect additional stats
                    additional_values = []
                    for add_stat in selected_additional_stats:
                        add_value = player_data.iloc[0][add_stat] if add_stat in player_data.columns else "N/A"
                        # Format the value if needed
                        if pd.isna(add_value):
                            add_value = "N/A"
                        elif add_stat in two_decimal_stats:
                            add_value = format_value(add_value)
                        additional_values.append(f"{stat_display_names.get(add_stat, add_stat)}: {add_value}")
                    additional_stats_values1.append(additional_values)
    if not years:
        st.info(f"No data found for player '{player1}' for the selected stat.")
        return

    # Collect data for player 2 if selected
    if player2:
        stat_values2 = []
        additional_stats_values2 = []
        # Using the same 'years' list to ensure alignment
        for sheet in years:
            df = session_state.excel_file.parse(sheet_name=sheet)
            df.columns = df.columns.str.strip()
            if 'Player' in df.columns and stat in df.columns:
                player_data = df[df['Player'] == player2]
                if not player_data.empty:
                    value = player_data.iloc[0][stat]
                    if pd.isna(value):
                        value = np.nan  # Use NaN to maintain alignment
                    else:
                        try:
                            value = float(value)
                            if stat in two_decimal_stats:
                                value = float(f"{value:.2f}")
                        except (ValueError, TypeError):
                            value = np.nan  # Use NaN for invalid values
                    stat_values2.append(value)
                    # Collect additional stats
                    additional_values = []
                    for add_stat in selected_additional_stats:
                        add_value = player_data.iloc[0][add_stat] if add_stat in player_data.columns else "N/A"
                        # Format the value if needed
                        if pd.isna(add_value):
                            add_value = "N/A"
                        elif add_stat in two_decimal_stats:
                            add_value = format_value(add_value)
                        additional_values.append(f"{stat_display_names.get(add_stat, add_stat)}: {add_value}")
                    additional_stats_values2.append(additional_values)
                else:
                    # If player data is missing for the year, append NaN
                    stat_values2.append(np.nan)
                    additional_stats_values2.append([])
    # Generate the plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))

    x_indices = np.arange(len(years))  # the label locations
    width = 0.35  # the width of the bars

    # Plot bars for player 1
    rects1 = ax.bar(x_indices - width/2, stat_values1, width, label=player1, color=bar_color)

    # Plot bars for player 2 if selected
    if player2:
        rects2 = ax.bar(x_indices + width/2, stat_values2, width, label=player2, color='orange')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(stat_display_names.get(stat, stat), fontsize=14)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(years, rotation=45, fontsize=12)

    # Use custom heading if provided
    if chart_heading:
        ax.set_title(chart_heading, fontweight='bold', fontsize=16)
    else:
        ax.set_title(f"{stat_display_names.get(stat, stat)} Over the Years", fontweight='bold', fontsize=16)

    ax.legend(fontsize=12)

    max_stat_value = max(stat_values1 + (stat_values2 if player2 else [])) if stat_values1 else 0

    # Adjust y-limit to make space for annotations
    ax.set_ylim(0, max_stat_value * 1.25)

    # Annotate bars with main stat value and additional stats
    def annotate_bars(rects, stat_values, additional_stats_values):
        for rect, main_value, add_values in zip(rects, stat_values, additional_stats_values):
            height = rect.get_height()
            y_offset = height + (max_stat_value * 0.01)
            # Annotate main value
            ax.text(rect.get_x() + rect.get_width()/2., y_offset,
                    f'{main_value}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            y_offset += (max_stat_value * 0.03)
            # Annotate additional stats
            for add_value in add_values:
                ax.text(rect.get_x() + rect.get_width()/2., y_offset,
                        f'({add_value})', ha='center', va='bottom', fontsize=8)
                y_offset += (max_stat_value * 0.03)

    annotate_bars(rects1, stat_values1, additional_stats_values1)
    if player2:
        annotate_bars(rects2, stat_values2, additional_stats_values2)

    # Adjust layout to prevent clipping
    fig.tight_layout()

    st.pyplot(fig)

def top_players_over_years(session_state):
    st.markdown("<h2 style='color: #2c3e50; font-size:32px;'>Top Players Over Years</h2>", unsafe_allow_html=True)

    # Stat selection
    default_stats = [
        "Matches", "Innings", "Runs", "High", "Not Out", "Avg",
        "50s", "100s", "Ducks", "Catches", "Wickets", "C+RO+S",
        "Avg.1", "Econ", "SR", "Best", "Full Overs"
    ]
    stat_display_names = {
        'Avg': 'Batting Avg',
        'Avg.1': 'Bowling Avg',
        'Econ': 'Bowling Econ',
        'SR': 'Bowling SR',
        'C+RO+S': 'Catches+RunOuts+Stumpings'
    }
    stat_options = [stat for stat in default_stats if stat != 'Player']
    display_stat_options = [stat_display_names.get(stat, stat) for stat in stat_options]
    display_stat = st.selectbox("Select Stat:", display_stat_options)
    reverse_stat_display_names = {v: k for k, v in stat_display_names.items()}
    stat = reverse_stat_display_names.get(display_stat, display_stat)

    # Sort order selection
    order = st.radio("Select Order:", ["Most", "Least"])

    # Chart Heading
    chart_heading = st.text_input("Enter Chart Title:")

    # Title color selection
    title_color_name = st.selectbox("Select Title Color:", list(title_colors.keys()))
    title_color = title_colors[title_color_name]

    # Bar color selection
    bar_color_name = st.selectbox("Select Bar Color:", list(bar_colors.keys()))
    bar_color = bar_colors[bar_color_name]

    # Additional stats selection
    selected_additional_stats = st.multiselect("Select Additional Stats to Display:", display_stat_options)
    selected_additional_stats = [reverse_stat_display_names.get(s, s) for s in selected_additional_stats]

    generate_button = st.button("Generate Chart")
    if generate_button:
        generate_top_players_chart(session_state, stat, order, selected_additional_stats,
                                   chart_heading, title_color, bar_color, stat_display_names)

def generate_top_players_chart(session_state, stat, order, selected_additional_stats,
                               chart_heading, title_color, bar_color, stat_display_names):
    two_decimal_stats = {'Avg', 'Avg.1', 'Econ', 'SR'}

    # Collect data across years
    years = []
    stat_values = []
    player_names = []
    additional_stats_values = []

    for sheet in session_state.excel_sheets:
        if sheet != "ALL TIME" and sheet != "Career Achievements":
            df = session_state.excel_file.parse(sheet_name=sheet)
            df.columns = df.columns.str.strip()

            # Check if required columns exist
            if 'Player' not in df.columns or stat not in df.columns:
                continue

            # Filter columns that exist in the DataFrame
            available_additional_stats = [s for s in selected_additional_stats if s in df.columns]

            # Create filtered DataFrame with only available columns
            columns_to_use = ['Player', stat] + available_additional_stats
            df_filtered = df[columns_to_use].dropna(subset=[stat])

            if df_filtered.empty:
                continue

            # Convert stat to numeric
            df_filtered[stat] = pd.to_numeric(df_filtered[stat], errors='coerce')
            df_filtered.dropna(subset=[stat], inplace=True)
            if df_filtered.empty:
                continue

            # Find the top player based on the stat
            if order == "Most":
                idx = df_filtered[stat].idxmax()
            else:
                idx = df_filtered[stat].idxmin()

            top_player = df_filtered.loc[idx, 'Player']
            top_value = df_filtered.loc[idx, stat]
            if pd.isna(top_value):
                continue  # Skip if top_value is NaN
            years.append(sheet)
            top_value = format_value(top_value) if stat in two_decimal_stats else top_value
            stat_values.append(top_value)
            player_names.append(top_player)

            # Collect additional stats
            additional_values = []
            for add_stat in available_additional_stats:
                add_value = df_filtered.loc[idx, add_stat]
                # Format the value if needed
                if pd.isna(add_value):
                    add_value = "N/A"
                elif add_stat in two_decimal_stats:
                    add_value = format_value(add_value)
                additional_values.append(f"{stat_display_names.get(add_stat, add_stat)}: {add_value}")
            additional_stats_values.append(additional_values)

    # Generate the plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(years, stat_values, color=bar_color)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(stat_display_names.get(stat, stat), fontsize=14)
    # Use custom heading if provided
    if chart_heading:
        ax.set_title(chart_heading, fontweight='bold', color=title_color, fontsize=16)
    else:
        ax.set_title(f"Top Players by {stat_display_names.get(stat, stat)} ({order}) Over the Years", fontweight='bold', color=title_color, fontsize=16)
    ax.tick_params(axis='x', rotation=45, labelsize=12)

    # Convert stat_values to numeric for y-axis scaling
    numeric_stat_values = []
    for val in stat_values:
        try:
            numeric_stat_values.append(float(val))
        except ValueError:
            numeric_stat_values.append(0)
    max_stat_value = max(numeric_stat_values) if numeric_stat_values else 0

    # Adjust y-limit to make space for annotations
    ax.set_ylim(0, max_stat_value * 1.25)

    # Annotate bars with player name, main stat value, and additional stats
    for bar, main_value, name, add_values in zip(bars, stat_values, player_names, additional_stats_values):
        height = bar.get_height()
        y_offset = height + (max_stat_value * 0.01)
        # Annotate player name
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{name}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        y_offset += (max_stat_value * 0.03)
        # Annotate main value
        ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                f'{main_value}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        y_offset += (max_stat_value * 0.03)
        # Annotate additional stats
        for add_value in add_values:
            ax.text(bar.get_x() + bar.get_width()/2., y_offset,
                    f'({add_value})', ha='center', va='bottom', fontsize=8)
            y_offset += (max_stat_value * 0.03)

    # Adjust layout to prevent clipping of tick-labels
    fig.tight_layout()

    st.pyplot(fig)

if __name__ == "__main__":
    main()