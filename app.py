import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

st.set_page_config(page_title="Calculadora de Opciones", layout="wide")

st.title("📊 Calculadora de Opciones y Valoración (Black-Scholes)")

# --- BARRA LATERAL PARA INGRESAR DATOS ---
st.sidebar.header("Parámetros de Entrada")
S = st.sidebar.number_input("Precio Subyacente (S)", value=100.0, step=1.0)
K = st.sidebar.number_input("Precio de Ejercicio (K)", value=100.0, step=1.0)
T_dias = st.sidebar.number_input("Días al Vencimiento", value=30.0, step=1.0)
r_pct = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=5.0, step=0.5)
sigma_pct = st.sidebar.number_input("Volatilidad Implícita (%)", value=20.0, step=1.0)

T = T_dias / 365.0
r = r_pct / 100.0
sigma = sigma_pct / 100.0

# --- CÁLCULO BLACK-SCHOLES Y GRIEGAS ---
def calcular_black_scholes(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
    theta_call = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    theta_put = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
    rho_call = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100.0
    rho_put = (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100.0
    
    return call, put, delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

c, p, d_c, d_p, gam, veg, th_c, th_p, rh_c, rh_p = calcular_black_scholes(S, K, T, r, sigma)

# --- VALORES EN PANTALLA ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Opciones CALL")
    st.metric("Precio CALL", f"${c:.2f}")
    st.write(f"**Delta:** {d_c:.4f} | **Theta:** {th_c:.4f} | **Rho:** {rh_c:.4f}")

with col2:
    st.subheader("Opciones PUT")
    st.metric("Precio PUT", f"${p:.2f}")
    st.write(f"**Delta:** {d_p:.4f} | **Theta:** {th_p:.4f} | **Rho:** {rh_p:.4f}")

st.write(f"**Gamma:** {gam:.4f} | **Vega:** {veg:.4f}")
st.markdown("---")

# --- GRÁFICOS ---
st.header("📈 Gráficos de Payoff y Griegas")

s_rango = np.linspace(S * 0.5, S * 1.5, 100)

# 1. Payoff Call
fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.plot(s_rango, np.maximum(s_rango - K, 0) - c, label="Call Payoff", color="green")
ax1.axhline(0, color="black", linestyle="--")
ax1.set_title("Payoff CALL al Vencimiento")
ax1.grid(True)
st.pyplot(fig1)
plt.close(fig1)

# 2. Payoff Put
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.plot(s_rango, np.maximum(K - s_rango, 0) - p, label="Put Payoff", color="red")
ax2.axhline(0, color="black", linestyle="--")
ax2.set_title("Payoff PUT al Vencimiento")
ax2.grid(True)
st.pyplot(fig2)
plt.close(fig2)

# 3. Delta
fig3, ax3 = plt.subplots(figsize=(6, 3))
deltas_c = [norm.cdf((math.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))) for s in s_rango]
ax3.plot(s_rango, deltas_c, color="blue", label="Delta Call")
ax3.set_title("Sensibilidad: Delta vs Precio Subyacente")
ax3.grid(True)
st.pyplot(fig3)
plt.close(fig3)

# 4. Gamma
fig4, ax4 = plt.subplots(figsize=(6, 3))
gammas = [norm.pdf((math.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))/(s*sigma*math.sqrt(T)) for s in s_rango]
ax4.plot(s_rango, gammas, color="purple", label="Gamma")
ax4.set_title("Sensibilidad: Gamma vs Precio Subyacente")
ax4.grid(True)
st.pyplot(fig4)
plt.close(fig4)

# 5. Vega
fig5, ax5 = plt.subplots(figsize=(6, 3))
vegas = [s * norm.pdf((math.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))*math.sqrt(T)/100.0 for s in s_rango]
ax5.plot(s_rango, vegas, color="orange", label="Vega")
ax5.set_title("Sensibilidad: Vega vs Precio Subyacente")
ax5.grid(True)
st.pyplot(fig5)
plt.close(fig5)

# 6. Theta
fig6, ax6 = plt.subplots(figsize=(6, 3))
thetas = [(- (s * norm.pdf((math.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf((math.log(s/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)) - sigma*math.sqrt(T))) / 365.0 for s in s_rango]
ax6.plot(s_rango, thetas, color="brown", label="Theta Call")
ax6.set_title("Sensibilidad: Theta vs Precio Subyacente")
ax6.grid(True)
st.pyplot(fig6)
plt.close(fig6)
