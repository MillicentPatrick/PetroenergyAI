import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import yfinance as yf
from pipeline import initialize_forecaster, initialize_maintenance_model
from fpdf import FPDF
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="PetroEnergy Dashboard", layout="wide")

def fetch_yfinance_data():
    """Fetch oil price data from Yahoo Finance with error handling"""
    try:
        df = yf.download(['CL=F', 'BZ=F'], period="1y", interval="1d", progress=False)
        df = df['Close'].copy()
        
        if 'CL=F' not in df.columns or 'BZ=F' not in df.columns:
            raise ValueError("Missing WTI or Brent price data from yfinance.")

        df = df.rename(columns={'CL=F': 'WTIPRICE', 'BZ=F': 'BRENTPRICE'}).reset_index()
        df = df.dropna(subset=['WTIPRICE', 'BRENTPRICE'])

        # Data processing
        df['DATE'] = pd.to_datetime(df['Date'])
        df.drop(columns=['Date'], inplace=True)
        df['EVENTTYPE'] = 'normal'
        df['EVENTIMPACT'] = 0.0
        df['WEATHERIMPACT'] = 0.0
        df['INVENTORYLEVEL'] = df['WTIPRICE'].rolling(5).mean().bfill()
        df['DEMANDFORECAST'] = df['BRENTPRICE'].rolling(5).mean().bfill()
        
        return df
    except Exception as e:
        st.error(f"Failed to fetch market data: {str(e)}")
        return pd.DataFrame()

def update_market_data_file():
    """Update market data file with new records"""
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'data', 'market_data.csv')
        live_data = fetch_yfinance_data()
        
        if live_data.empty:
            return
            
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path, parse_dates=['DATE'])
            latest = existing['DATE'].max()
            new_rows = live_data[live_data['DATE'] > latest]
            
            if not new_rows.empty:
                updated = pd.concat([existing, new_rows], ignore_index=True)
                updated.to_csv(file_path, index=False)
                st.success("Market data updated successfully!")
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            live_data.to_csv(file_path, index=False)
    except Exception as e:
        st.error(f"Error updating market data: {str(e)}")

def generate_pdf(summary_text):
    """Generate PDF from summary text with proper formatting"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Set document properties
        pdf.set_title("PetroEnergy Executive Summary")
        pdf.set_author("PetroEnergy Analytics")
        
        # Set margins (left, top, right)
        pdf.set_margins(15, 15, 15)
        
        # Calculate effective width
        effective_width = pdf.w - 30
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(effective_width, 10, "PetroEnergy Executive Summary", ln=True, align='C')
        pdf.ln(10)
        
        # Set body font
        pdf.set_font("Arial", size=12)
        
        # Process each paragraph
        paragraphs = summary_text.split('\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                if para.startswith('**') and para.endswith('**'):
                    pdf.set_font("Arial", 'B', 12)
                    para = para[2:-2].strip()
                    pdf.multi_cell(effective_width, 8, para)
                    pdf.set_font("Arial", '', 12)
                else:
                    pdf.multi_cell(effective_width, 8, para)
                pdf.ln(5)
        
        # Add footer
        pdf.set_y(-15)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"Generated on {date.today().strftime('%Y-%m-%d')}", 0, 0, 'C')
        
        return BytesIO(pdf.output(dest='S').encode('latin-1'))
    except Exception as e:
        st.error(f"Failed to generate PDF: {str(e)}")
        return BytesIO()

def load_data(file_name):
    """Helper function to load data with error handling"""
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'data', file_name)
        return pd.read_csv(file_path, parse_dates=True)
    except Exception as e:
        st.error(f"Failed to load {file_name}: {str(e)}")
        return pd.DataFrame()

def main():
    # Initialize session state
    if 'last_updated' not in st.session_state:
        st.session_state['last_updated'] = date.today() - timedelta(days=8)  # Force initial update
    
    # Auto-update once a week
    if (date.today() - st.session_state['last_updated']).days >= 7:
        with st.spinner("Updating market data..."):
            update_market_data_file()
            st.session_state['last_updated'] = date.today()

    st.title("PetroEnergy Solutions - Business Intelligence Dashboard")

    # Refresh button
    if st.button("🔄 Refresh Live Data"):
        with st.spinner("Refreshing data..."):
            update_market_data_file()

    # Load data with error handling
    production_data = load_data('production_data.csv')
    equipment_data = load_data('equipment_data.csv')
    market_data = load_data('market_data.csv')

    # Check if data loaded successfully
    if production_data.empty or equipment_data.empty or market_data.empty:
        st.warning("Some data failed to load. Please check your data files.")
        return

    # Clean market data
    market_data = market_data.dropna(subset=['WTIPRICE', 'BRENTPRICE'])

    # Initialize models
    forecaster = initialize_forecaster(market_data)
    maintenance_model = initialize_maintenance_model(equipment_data)

    # Sidebar controls
    st.sidebar.header("Controls")
    selected_facility = st.sidebar.selectbox(
        "Select Facility", 
        production_data['FACILITYID'].unique()
    )
    date_range = st.sidebar.date_input(
        "Date Range", 
        [date.today() - timedelta(days=30), date.today()]
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Production", "Market Forecast", "Equipment Health", 
        "Sensor Readings", "Summary Report"
    ])

    with tab1:
        st.subheader("Production Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                production_data[production_data['FACILITYID'] == selected_facility],
                x='DATE', y='PRODUCTIONVOLUME',
                title=f"Production Volume - {selected_facility}"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            status_counts = production_data['STATUS'].value_counts().reset_index()
            status_counts.columns = ['STATUS', 'count']
            fig = px.pie(
                status_counts, 
                values='count', 
                names='STATUS', 
                title="Facility Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("30-Day Oil Price Forecast")
        future_dates = pd.date_range(start=date.today(), periods=30)
        price_preds = forecaster.predict_prices(future_dates)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_preds['DATE'], 
            y=price_preds['WTIPRICE_PRED'],
            name='WTI Price Forecast', 
            line=dict(color='blue')
        )
        fig.add_trace(go.Scatter(
            x=price_preds['DATE'], 
            y=price_preds['BRENTPRICE_PRED'],
            name='Brent Price Forecast', 
            line=dict(color='green')
        )
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Price ($)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Equipment Health Monitoring")
        facility_equipment = equipment_data[
            equipment_data['FACILITYID'] == selected_facility
        ].copy()
        
        anomalies = maintenance_model.predict_anomalies(facility_equipment)
        facility_equipment['anomaly'] = anomalies.map({1: 'Normal', -1: 'Anomaly'})
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                facility_equipment, 
                x='TIMESTAMP', 
                y='HEALTHSCORE',
                color='EQUIPMENTID', 
                title="Health Scores Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            counts = facility_equipment['anomaly'].value_counts().reset_index()
            counts.columns = ['anomaly', 'count']
            fig = px.bar(
                counts, 
                x='anomaly', 
                y='count', 
                color='anomaly',
                color_discrete_map={'Normal': 'green', 'Anomaly': 'red'},
                title="Anomaly Detection Results"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Maintenance Report
        maintenance_report = maintenance_model.generate_maintenance_report(equipment_data)
        st.subheader("Top 5 Maintenance Priorities")
        st.dataframe(maintenance_report.head())
        
        st.download_button(
            "📥 Download Maintenance CSV", 
            data=maintenance_report.to_csv(index=False), 
            file_name="weekly_maintenance_schedule.csv",
            mime='text/csv'
        )

    with tab4:
        st.subheader("Simulated Sensor Readings")
        sensors = pd.DataFrame({
            'Sensor': ['Temp A', 'Vibration B', 'Pressure C'],
            'Reading': [round(x, 2) for x in [75 + 5, 0.9 + 0.1, 30 + 1.5]],
            'Status': ['Normal', 'Normal', 'Warning']
        })
        st.dataframe(sensors)

    with tab5:
        st.subheader("Executive Summary")
        
        # Calculate metrics
        avg_prod = production_data['PRODUCTIONVOLUME'].mean()
        last_health = facility_equipment['HEALTHSCORE'].iloc[-1] if not facility_equipment.empty else 'N/A'
        price_trend = "increasing" if price_preds['WTIPRICE_PRED'].iloc[-1] > price_preds['WTIPRICE_PRED'].iloc[0] else "decreasing"
        anomaly_count = len(anomalies[anomalies == -1])
        
        # Alert system
        if anomaly_count > 0:
            st.warning(f" {anomaly_count} anomalies found. Consider maintenance.")
        else:
            st.success(" All equipment normal.")
        
        # Generate summary
        summary = f"""
        **Production Overview**: Average production across facilities is {avg_prod:,.2f} units. 
        The selected facility {selected_facility} shows typical operational patterns.

        **Equipment Health**: Latest equipment health score is {last_health:.2f}, with {anomaly_count} 
        potential anomalies detected in the last 30 days.

        **Market Outlook**: Oil prices are forecasted to be {price_trend} over the next 30 days, 
        with WTI reaching ${price_preds['WTIPRICE_PRED'].iloc[-1]:.2f} and Brent at ${price_preds['BRENTPRICE_PRED'].iloc[-1]:.2f}.

        **Recommendations**: {'Consider preventive maintenance for flagged equipment.' if anomaly_count > 0 else 'No immediate maintenance concerns.'}
        {' Adjust production schedules to take advantage of rising prices.' if price_trend == 'increasing' else ''}
        """
        
        st.markdown(summary)

        # Download options
        csv_summary = pd.DataFrame({
            'Metric': ['Avg Production', 'Latest Health Score', 'WTI Forecast', 'Brent Forecast'],
            'Value': [avg_prod, last_health, price_preds['WTIPRICE_PRED'].iloc[-1], price_preds['BRENTPRICE_PRED'].iloc[-1]]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Summary CSV", 
                data=csv_summary.to_csv(index=False), 
                file_name="summary_report.csv", 
                mime='text/csv'
            )
        with col2:
            st.download_button(
                "📄 Download Summary PDF", 
                data=generate_pdf(summary), 
                file_name="summary_report.pdf",
                mime='application/pdf'
            )

if __name__ == "__main__":
    main()
