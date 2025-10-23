import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title('Polynomial Curve Fitting')
st.write('Upload a CSV file with X and Y data to fit a polynomial curve')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Raw Data")
        st.dataframe(df.head())

        # Column selection
        st.subheader("Column Selection")
        # Clean column names (strip spaces) just in case
        df.columns = df.columns.str.strip()
        x_column = st.selectbox("Select X column", df.columns)
        y_column = st.selectbox("Select Y column", df.columns, index=min(1, len(df.columns) - 1))

        # Extract and clean data
        x_data = df[x_column].astype(float).values
        y_data = df[y_column].astype(float).values
        valid = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid]
        y_data = y_data[valid]

        # Degree selection
        degree = st.slider("Select polynomial degree", min_value=1, max_value=15, value=8)

        if st.button("Fit Curve"):
            # Fit polynomial using numpy.polyfit (coefficients returned highest degree first)
            coeffs = np.polyfit(x_data, y_data, degree)

            # Evaluate fit over a dense grid
            x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
            y_fit = np.polyval(coeffs, x_fit)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x_data, y_data, label='Data Points', alpha=0.7)
            ax.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
            ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title('Curve Fitting')
            ax.grid(True)

            st.pyplot(fig)

            # Display coefficients (highest degree first)
            st.subheader("Fitted Polynomial Coefficients (highest degree first)")
            powers = list(range(degree, -1, -1))
            coeffs_df = pd.DataFrame({
                'Power': powers,
                'Coefficient': coeffs
            })
            st.dataframe(coeffs_df)

            # Display polynomial equation (highest degree first)
            terms = []
            for i, c in enumerate(coeffs):
                power = degree - i
                if power == 0:
                    terms.append(f"({c:.3e})")
                elif power == 1:
                    terms.append(f"({c:.3e})x")
                else:
                    terms.append(f"({c:.3e})x^{power}")
            
            # Join with " + " and clean up " + -" for negative coefficients
            equation = " + ".join(terms).replace("+ -", "- ")

            st.subheader("Polynomial Equation")
            st.write(f"y = {equation}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get started")

# Add instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a CSV file containing your data
2. Select X and Y columns from your data
3. Adjust the polynomial degree using the slider
4. Click 'Fit Curve' to generate the polynomial fit
""")

