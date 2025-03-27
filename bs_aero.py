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
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Raw Data")
        st.dataframe(df.head())
        
        st.subheader("Column Selection")
        x_column = st.selectbox("Select X column", df.columns)
        y_column = st.selectbox("Select Y column", df.columns, index=min(1, len(df.columns)-1))
        
        x_data = df[x_column].values
        y_data = df[y_column].values
        
        degree = st.slider("Select polynomial degree", min_value=1, max_value=15, value=8)
        
        if st.button("Fit Curve"):
            coefficients = fit_curve(x_data, y_data, degree)
        
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = polynomial(x_fit, *coefficients)
        
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x_data, y_data, label='Data Points')
            ax.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
            ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title('Curve Fitting')
        
            st.pyplot(fig)
            
            st.subheader("Fitted Polynomial Coefficients")
            coeffs_df = pd.DataFrame({
                'Power': range(len(coefficients)),
                'Coefficient': coefficients
            })
            st.dataframe(coeffs_df)
            
            equation = " + ".join([f"{coef:.6f}x^{i}" if i > 0 else f"{coef:.6f}" 
                                 for i, coef in enumerate(coefficients)])
            st.subheader("Polynomial Equation")
            st.write(f"y = {equation}")
        
            desmos_url = f"https://www.desmos.com/calculator?lang=en&y={','.join(map(str, coefficients))}"
            st.subheader("Desmos Graph")
            st.write(f'<iframe src="{desmos_url}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get started")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a CSV file containing your data
2. Select X and Y columns from your data
3. Adjust the polynomial degree using the slider
4. Click 'Fit Curve' to generate the polynomial fit
""")
