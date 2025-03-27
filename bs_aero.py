import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import streamlit as st
import io

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def fit_curve(x_data, y_data, degree=6):
    initial_guess = np.ones(degree + 1) 
    params, _ = curve_fit(lambda x, *p: polynomial(x, *p), x_data, y_data, p0=initial_guess)
    return params

st.title('Polynomial Curve Fitting')
st.write('Upload a CSV file with X and Y data to fit a polynomial curve')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Display the raw data
        st.subheader("Raw Data")
        st.dataframe(df.head())
        
        # Select columns
        st.subheader("Column Selection")
        x_column = st.selectbox("Select X column", df.columns)
        y_column = st.selectbox("Select Y column", df.columns, index=min(1, len(df.columns)-1))
        
        # Get data from selected columns
        x_data = df[x_column].values
        y_data = df[y_column].values
        
        # Polynomial degree slider
        degree = st.slider("Select polynomial degree", min_value=1, max_value=15, value=8)
        
        # Fit curve and plot
        if st.button("Fit Curve"):
            # Fit the curve
            coefficients = fit_curve(x_data, y_data, degree)
            
            # Generate points for the fitted curve
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = polynomial(x_fit, *coefficients)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x_data, y_data, label='Data Points')
            ax.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
            ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title('Curve Fitting')
            
            # Display plot in Streamlit
            st.pyplot(fig)
            
            # Display coefficients
            st.subheader("Fitted Polynomial Coefficients")
            coeffs_df = pd.DataFrame({
                'Power': range(len(coefficients)),
                'Coefficient': coefficients
            })
            st.dataframe(coeffs_df)
            
            # Display polynomial equation
            equation = " + ".join([f"{coef:.6f}x^{i}" if i > 0 else f"{coef:.6f}" 
                                 for i, coef in enumerate(coefficients)])
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