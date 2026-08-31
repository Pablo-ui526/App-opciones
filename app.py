import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

st.set_page_config(page_title="Estrategia de Opciones", layout="wide")

st.title("📊 Análisis de Opciones: Estrategia, Puntos de Equilibrio y Griegas")

# --- BARRA LATERAL: PARÁMETROS GENERALES ---
st.sidebar.header("⚙️ Parámetros Generales")
S = st.sidebar.number_input("Precio Subyacente Actual (S)", value=100.0, step=1.0)
T_dias = st.sidebar.number_input("Días al Vencimiento", value=30.0, step=1.0)
r_pct = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=5.0, step=0.5)
sigma_pct = st.sidebar.number_input("Volatilidad Implícita (%)", value=20.0, step=1.0)

T = max(T_dias / 365.0, 0.0001)
r = r_pct / 100.0
sigma = sigma_pct / 100.0

# --- CÁLCULO BLACK-SCHOLES ---
def bs_precio_griegas(S, K, T, r, sigma, tipo="Call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if tipo == "Call":
        precio = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        precio = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
        
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
    
    return precio, delta, gamma, vega, theta

# --- INGRESO DE HASTA 4 PATAS / CONTRATOS ---
st.subheader("📋 Configuración de las 4 Patas de la Estrategia")

cols = st.columns(4)
patas = []

for i in range(4):
    with cols[i]:
        st.markdown(f"### Pata {i+1}")
        activo = st.checkbox(f"Activar Pata {i+1}", value=(i < 2), key=f"act_{i}")
        tipo = st.selectbox(f"Tipo", ["Call", "Put"], key=f"tipo_{i}")
        posicion = st.selectbox(f"Posición", ["Comprado (Long)", "Vendido (Short)"], key=f"pos_{i}")
        K_pata = st.number_input(f"Strike (K)", value=100.0 + (i-1)*5, step=1.0, key=f"k_{i}")
        prima_mercado = st.number_input(f"Prima de Mercado", value=2.5, step=0.1, key=f"p_{i}")
        cant = st.number_input(f"Cantidad", value=1, step=1, key=f"cant_{i}")
        
        if activo:
            bs_teorico, d, g, v, th = bs_precio_griegas(S, K_pata, T, r, sigma, tipo)
            
            # Diagnóstico Cara / Barata
            diferencia = prima_mercado - bs_teorico
            if abs(diferencia) < 0.1:
                estado = "⚖️ Precio Justo"
            elif diferencia > 0:
                estado = "🔴 CARA (Conviene Vendela)"
            else:
                estado = "🟢 BARATA (Conviene Comprar)"
                
            st.caption(f"BS Teórico: **${bs_teorico:.2f}**")
            st.caption(f"Diagnóstico: **{estado}**")
            
            patas.append({
                "tipo": tipo,
                "posicion": posicion,
                "K": K_pata,
                "prima": prima_mercado,
                "cant": cant,
                "delta": d if "Comprado" in posicion else -d,
                "gamma": g if "Comprado" in posicion else -g,
                "vega": v if "Comprado" in posicion else -v,
                "theta": th if "Comprado" in posicion else -th,
            })

st.markdown("---")

# --- RESUMEN CONSOLIDADO ---
total_delta = sum(p["delta"] * p["cant"] for p in patas)
total_gamma = sum(p["gamma"] * p["cant"] for p in patas)
total_vega = sum(p["vega"] * p["cant"] for p in patas)
total_theta = sum(p["theta"] * p["cant"] for p in patas)

st.subheader("📊 Totales Consolidados de la Estrategia")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Delta Total", f"{total_delta:.4f}")
m2.metric("Gamma Total", f"{total_gamma:.4f}")
m3.metric("Vega Total", f"{total_vega:.4f}")
m4.metric("Theta Total", f"{total_theta:.4f}")

# --- CÁLCULO DE PAYOFFS ---
s_rango = np.linspace(S * 0.5, S * 1.5, 200)

def calcular_payoff_pata(s_rango, pata):
    K = pata["K"]
    prima = pata["prima"]
    cant = pata["cant"]
    
    if pata["tipo"] == "Call":
        payoff_unitario = np.maximum(s_rango - K, 0) - prima
    else:
        payoff_unitario = np.maximum(K - s_rango, 0) - prima
        
    if "Vendido" in pata["posicion"]:
        payoff_unitario = -payoff_unitario
        
    return payoff_unitario * cant

payoffs_patas = [calcular_payoff_pata(s_rango, p) for p in patas]
payoff_total = sum(payoffs_patas) if len(payoffs_patas) > 0 else np.zeros_like(s_rango)

# --- 1. GRÁFICO PRINCIPAL CONSOLIDADO (SUMADO) CON ZONAS ---
st.markdown("---")
st.header("🎯 Gráfico Consolidado: Payoff Total con Zonas y Puntos de Equilibrio")

fig_tot, ax_tot = plt.subplots(figsize=(10, 4.5))

# Trazar Payoff Total
ax_tot.plot(s_rango, payoff_total, label="Payoff Total Estrategia", color="black", linewidth=2.5)
ax_tot.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax_tot.axvline(S, color="blue", linestyle=":", label=f"S Actual (${S:.2f})")

# Relleno verde (Ganancia) y rojo (Pérdida)
ax_tot.fill_between(s_rango, payoff_total, 0, where=(payoff_total >= 0), color="green", alpha=0.25, label="Zona Ganancia (Conviene)")
ax_tot.fill_between(s_rango, payoff_total, 0, where=(payoff_total < 0), color="red", alpha=0.25, label="Zona Pérdida")

# Marcación de Puntos de Equilibrio (Break-even)
zero_crossings = np.where(np.diff(np.sign(payoff_total)))[0]
for idx in zero_crossings:
    be_price = s_rango[idx]
    ax_tot.plot(be_price, 0, 'ro', markersize=8)
    ax_tot.annotate(f"B.E. ${be_price:.1f}", (be_price, 0), textcoords="offset points", xytext=(0,10), ha='center', weight='bold')

ax_tot.set_title("Estrategia Sumada: Zonas de Ganancia / Pérdida y Puntos de Equilibrio")
ax_tot.set_xlabel("Precio Subyacente al Vencimiento")
ax_tot.set_ylabel("Ganancia / Pérdida ($)")
ax_tot.grid(True, alpha=0.3)
ax_tot.legend()

st.pyplot(fig_tot)
plt.close(fig_tot)

# --- 2. LOS 6 GRÁFICOS SEPARADOS ---
st.markdown("---")
st.header("📈 6 Gráficos Separados por Pata y Sensibilidades")

g1, g2 = st.columns(2)

with g1:
    # Gráfico 1: Payoff Pata 1
    if len(patas) >= 1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(s_rango, payoffs_patas[0], color="darkgreen")
        ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax1.fill_between(s_rango, payoffs_patas[0], 0, where=(payoffs_patas[0]>=0), color="green", alpha=0.2)
        ax1.fill_between(s_rango, payoffs_patas[0], 0, where=(payoffs_patas[0]<0), color="red", alpha=0.2)
        ax1.set_title("1. Payoff Pata 1 (Individual)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    # Gráfico 3: Payoff Pata 3
    if len(patas) >= 3:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(s_rango, payoffs_patas[2], color="teal")
        ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax3.fill_between(s_rango, payoffs_patas[2], 0, where=(payoffs_patas[2]>=0), color="green", alpha=0.2)
        ax3.fill_between(s_rango, payoffs_patas[2], 0, where=(payoffs_patas[2]<0), color="red", alpha=0.2)
        ax3.set_title("3. Payoff Pata 3 (Individual)")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    # Gráfico 5: Delta Curva
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    deltas_curva = [sum((norm.cdf((math.log(s/p["K"])+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))) if p["tipo"]=="Call" else norm.cdf((math.log(s/p["K"])+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))-1.0) * (1 if "Comprado" in p["posicion"] else -1) * p["cant"] for p in patas) for s in s_rango]
    ax5.plot(s_rango, deltas_curva, color="blue")
    ax5.set_title("5. Delta Consolidado vs Precio")
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    plt.close(fig5)

with g2:
    # Gráfico 2: Payoff Pata 2
    if len(patas) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(s_rango, payoffs_patas[1], color="darkred")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.fill_between(s_rango, payoffs_patas[1], 0, where=(payoffs_patas[1]>=0), color="green", alpha=0.2)
        ax2.fill_between(s_rango, payoffs_patas[1], 0, where=(payoffs_patas[1]<0), color="red", alpha=0.2)
        ax2.set_title("2. Payoff Pata 2 (Individual)")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    # Gráfico 4: Payoff Pata 4
    if len(patas) >= 4:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.plot(s_rango, payoffs_patas[3], color="purple")
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax4.fill_between(s_rango, payoffs_patas[3], 0, where=(payoffs_patas[3]>=0), color="green", alpha=0.2)
        ax4.fill_between(s_rango, payoffs_patas[3], 0, where=(payoffs_patas[3]<0), color="red", alpha=0.2)
        ax4.set_title("4. Payoff Pata 4 (Individual)")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    # Gráfico 6: Gamma Curva
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    gammas_curva = [sum((norm.pdf((math.log(s/p["K"])+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))/(s*sigma*math.sqrt(T))) * (1 if "Comprado" in p["posicion"] else -1) * p["cant"] for p in patas) for s in s_rango]
    ax6.plot(s_rango, gammas_curva, color="orange")
    ax6.set_title("6. Gamma Consolidado vs Precio")
    ax6.grid(True, alpha=0.3)
    st.pyplot(fig6)
    plt.close(fig6)
