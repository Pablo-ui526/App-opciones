import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <--- Esta es la única línea clave que agregamos arriba
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- CÁLCULO BLACK-SCHOLES Y GRIEGAS ---
def calcular_black_scholes(S, K, T_dias, r, sigma, tipo):
    T = T_dias / 365.0
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0, 0, 0, 0, 0
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if tipo == "CALL":
        precio = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        precio = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100.0
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2 if tipo=="CALL" else -d2)) / 365.0

    return precio, delta, gamma, theta, vega

# --- BÚSQUEDA DE PUNTOS DE EQUILIBRIO ---
def obtener_puntos_equilibrio(precios, payoff):
    zero_crossings = np.where(np.diff(np.sign(payoff)))[0]
    puntos = []
    for idx in zero_crossings:
        x1, x2 = precios[idx], precios[idx+1]
        y1, y2 = payoff[idx], payoff[idx+1]
        if y2 != y1:
            x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
            puntos.append(x_zero)
    return puntos

# --- DATOS DE ENTRADA (Cambiá los valores acá) ---
# Formato: [Tipo, PrecioAccion, Strike, DiasVenc, TasaAnual%, Volatilidad%, PrimaMercado]
p_cA = {'tipo': "CALL", 'S': 6750, 'K': 6000, 'T': 30, 'r': 0.40, 'sig': 0.50, 'p': 600} # Compra A
p_vA = {'tipo': "CALL", 'S': 6750, 'K': 6800, 'T': 30, 'r': 0.40, 'sig': 0.50, 'p': 860} # Venta A
p_cB = {'tipo': "PUT",  'S': 6750, 'K': 5800, 'T': 30, 'r': 0.40, 'sig': 0.50, 'p': 200} # Compra B
p_vB = {'tipo': "PUT",  'S': 6750, 'K': 6800, 'T': 30, 'r': 0.40, 'sig': 0.50, 'p': 354} # Venta B

# RESUMEN EN PANTALLA
costo_compra = p_cA['p'] + p_cB['p']
ingreso_venta = p_vA['p'] + p_vB['p']

print(f"=== RESUMEN DE ENTRADA ===")
print(f"SUMA COMPRA (A+B): Costo Total Entrada = ${costo_compra:.2f}")
print(f"SUMA VENTA  (A+B): Ingreso Total Cobrado = ${ingreso_venta:.2f}\n")

# CÁLCULOS Y GRÁFICOS
p_teor_cA, d_cA, g_cA, t_cA, v_cA = calcular_black_scholes(p_cA['S'], p_cA['K'], p_cA['T'], p_cA['r'], p_cA['sig'], p_cA['tipo'])
p_teor_cB, d_cB, g_cB, t_cB, v_cB = calcular_black_scholes(p_cB['S'], p_cB['K'], p_cB['T'], p_cB['r'], p_cB['sig'], p_cB['tipo'])
p_teor_vA, d_vA, g_vA, t_vA, v_vA = calcular_black_scholes(p_vA['S'], p_vA['K'], p_vA['T'], p_vA['r'], p_vA['sig'], p_vA['tipo'])
p_teor_vB, d_vB, g_vB, t_vB, v_vB = calcular_black_scholes(p_vB['S'], p_vB['K'], p_vB['T'], p_vB['r'], p_vB['sig'], p_vB['tipo'])

S_ref = p_cA['S']
precios = np.linspace(S_ref * 0.7, S_ref * 1.3, 150)

pay_cA = (np.maximum(precios - p_cA['K'], 0) if p_cA['tipo'] == "CALL" else np.maximum(p_cA['K'] - precios, 0)) - p_cA['p']
pay_cB = (np.maximum(precios - p_cB['K'], 0) if p_cB['tipo'] == "CALL" else np.maximum(p_cB['K'] - precios, 0)) - p_cB['p']
pay_vA = p_vA['p'] - (np.maximum(precios - p_vA['K'], 0) if p_vA['tipo'] == "CALL" else np.maximum(p_vA['K'] - precios, 0))
pay_vB = p_vB['p'] - (np.maximum(precios - p_vB['K'], 0) if p_vB['tipo'] == "CALL" else np.maximum(p_vB['K'] - precios, 0))

pay_sumC = pay_cA + pay_cB
pay_sumV = pay_vA + pay_vB

# EVALUACIÓN IA
teorico_sum_C = p_teor_cA + p_teor_cB
mercado_sum_C = p_cA['p'] + p_cB['p']
dif_C = ((mercado_sum_C - teorico_sum_C) / teorico_sum_C) * 100 if teorico_sum_C > 0 else 0

teorico_sum_V = p_teor_vA + p_teor_vB
mercado_sum_V = p_vA['p'] + p_vB['p']
dif_V = ((mercado_sum_V - teorico_sum_V) / teorico_sum_V) * 100 if teorico_sum_V > 0 else 0

diag_C = f"OPINIÓN IA: {'CARA' if dif_C>5 else 'BARATA' if dif_C<-5 else 'EN PRECIO'} ({dif_C:+.1f}%)\nMkt: ${mercado_sum_C:.0f} vs Teórico: ${teorico_sum_C:.0f}"
color_box_C = "#fadbd8" if dif_C > 5 else "#d4efdf" if dif_C < -5 else "#fcf3cf"

diag_V = f"OPINIÓN IA: {'CARA' if dif_V>5 else 'BARATA' if dif_V<-5 else 'EN PRECIO'} ({dif_V:+.1f}%)\nMkt: ${mercado_sum_V:.0f} vs Teórico: ${teorico_sum_V:.0f}"
color_box_V = "#d4efdf" if dif_V > 5 else "#fadbd8" if dif_V < -5 else "#fcf3cf"

def graficar_cuadro(ax, precios, payoff, color_linea, titulo, delta, gamma, theta, vega):
    ax.plot(precios, payoff, color=color_linea, linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.fill_between(precios, payoff, 0, where=(payoff >= 0), color='#abebc6', alpha=0.5, interpolate=True)
    ax.fill_between(precios, payoff, 0, where=(payoff < 0), color='#f9ebea', alpha=0.6, interpolate=True)

    pts_eq = obtener_puntos_equilibrio(precios, payoff)
    for pe in pts_eq:
        ax.axvline(pe, color='purple', linestyle=':', linewidth=1.2)
        ax.text(pe, 0, f" PE:{pe:.0f}", color='purple', fontsize=8, fontweight='bold', verticalalignment='bottom')

    ax.set_title(titulo, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    txt_g = f"Δ:{delta:+.2f} | Γ:{gamma:+.3f}\nΘ:${theta:+.1f}/d | Vega:${vega:+.1f}"
    ax.text(0.03, 0.93, txt_g, transform=ax.transAxes, fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

fig, axes = plt.subplots(3, 2, figsize=(11, 10))
graficar_cuadro(axes[0,0], precios, pay_cA, "green", f"1. POS COMPRA A ({p_cA['tipo']} {p_cA['K']:.0f})", d_cA, g_cA, t_cA, v_cA)
graficar_cuadro(axes[0,1], precios, pay_vA, "darkred", f"2. POS VENTA A ({p_vA['tipo']} {p_vA['K']:.0f})", -d_vA, -g_vA, -t_vA, -v_vA)
graficar_cuadro(axes[1,0], precios, pay_cB, "green", f"3. POS COMPRA B ({p_cB['tipo']} {p_cB['K']:.0f})", d_cB, g_cB, t_cB, v_cB)
graficar_cuadro(axes[1,1], precios, pay_vB, "darkred", f"4. POS VENTA B ({p_vB['tipo']} {p_vB['K']:.0f})", -d_vB, -g_vB, -t_vB, -v_vB)
graficar_cuadro(axes[2,0], precios, pay_sumC, "green", "5. SUMA RESULTADO A + B COMPRA", d_cA+d_cB, g_cA+g_cB, t_cA+t_cB, v_cA+v_cB)
graficar_cuadro(axes[2,1], precios, pay_sumV, "darkred", "6. SUMA RESULTADO A + B VENTA", -(d_vA+d_vB), -(g_vA+g_vB), -(t_vA+t_vB), -(v_vA+v_vB))

axes[2,0].text(0.98, 0.05, diag_C, transform=axes[2,0].transAxes, fontsize=7, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor=color_box_C, edgecolor='black', alpha=0.9))
axes[2,1].text(0.98, 0.05, diag_V, transform=axes[2,1].transAxes, fontsize=7, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor=color_box_V, edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.show()
