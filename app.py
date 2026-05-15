
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# APP: ECUACIONES ESTRUCTURALES EN ECONOMETRÍA
# Autor presentación: Juan Pablo Vargas
# Enfoque: modelos de ecuaciones simultáneas, identificación y 2SLS
# ============================================================

st.set_page_config(
    page_title="Ecuaciones estructurales | Econometría",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# ESTILO VISUAL
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
        padding: 2.2rem;
        border-radius: 26px;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.20);
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: 2.7rem;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin-bottom: 0.2rem;
    }

    .card {
        background: white;
        padding: 1.2rem 1.3rem;
        border-radius: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 1rem;
    }

    .mini-card {
        background: white;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
        border-left: 5px solid #2563eb;
        margin-bottom: 0.85rem;
    }

    .warning-card {
        background: #fff7ed;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border-left: 5px solid #f97316;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.85rem;
    }

    .success-card {
        background: #ecfdf5;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border-left: 5px solid #10b981;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.85rem;
    }

    .danger-card {
        background: #fef2f2;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border-left: 5px solid #ef4444;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.85rem;
    }

    .metric-box {
        background: white;
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.06);
    }

    .metric-box h3 {
        margin: 0;
        color: #0f172a;
    }

    .metric-box p {
        margin: 0.2rem 0 0 0;
        color: #475569;
        font-size: 0.92rem;
    }

    .eq-box {
        background: #0f172a;
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# FUNCIONES
# -----------------------------
@st.cache_data
def simular_oferta_demanda(n=500, seed=42, beta_precio_demanda=-1.2, alpha_precio_oferta=1.0):
    """
    Simula un sistema estructural simple de oferta y demanda.

    Demanda:
        Qd = a0 + a1*P + a2*Ingreso + u_d

    Oferta:
        Qs = b0 + b1*P + b2*Costo + u_s

    En equilibrio:
        Qd = Qs = Q
    """
    rng = np.random.default_rng(seed)

    ingreso = rng.normal(50, 10, n)
    costo = rng.normal(20, 5, n)

    u_d = rng.normal(0, 6, n)
    u_s = rng.normal(0, 6, n)

    a0 = 120
    a1 = beta_precio_demanda
    a2 = 1.5

    b0 = 20
    b1 = alpha_precio_oferta
    b2 = -1.1

    # Equilibrio:
    # a0 + a1P + a2Ingreso + u_d = b0 + b1P + b2Costo + u_s
    # P = (b0 - a0 + b2Costo - a2Ingreso + u_s - u_d) / (a1 - b1)
    precio = (b0 - a0 + b2 * costo - a2 * ingreso + u_s - u_d) / (a1 - b1)
    cantidad = a0 + a1 * precio + a2 * ingreso + u_d

    df = pd.DataFrame({
        "Precio": precio,
        "Cantidad": cantidad,
        "Ingreso": ingreso,
        "Costo": costo,
        "Shock_demanda": u_d,
        "Shock_oferta": u_s
    })

    return df


def ols(y, X):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    s2 = float((resid.T @ resid) / (n - k))
    var_beta = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)
    t = beta / se
    return beta.flatten(), se.flatten(), t.flatten(), yhat.flatten(), resid.flatten()


def estimar_modelos(df):
    """
    Modelo objetivo: demanda estructural
    Cantidad = beta0 + beta1*Precio + beta2*Ingreso + error

    Problema:
    Precio es endógeno porque se determina simultáneamente con cantidad.

    Instrumentos para Precio:
    Costo e Ingreso.
    En la ecuación de demanda, Costo es un instrumento natural porque desplaza oferta,
    afecta el precio de equilibrio, pero no debería afectar directamente la demanda.
    """

    y = df["Cantidad"].values
    precio = df["Precio"].values
    ingreso = df["Ingreso"].values
    costo = df["Costo"].values
    uno = np.ones(len(df))

    # OLS ingenuo
    X_ols = np.column_stack([uno, precio, ingreso])
    beta_ols, se_ols, t_ols, yhat_ols, resid_ols = ols(y, X_ols)

    # Primera etapa: Precio ~ Ingreso + Costo
    Z = np.column_stack([uno, ingreso, costo])
    gamma, se_gamma, t_gamma, precio_hat, resid_first = ols(precio, Z)

    # Segunda etapa: Cantidad ~ Precio_hat + Ingreso
    X_iv = np.column_stack([uno, precio_hat, ingreso])
    beta_iv, se_iv, t_iv, yhat_iv, resid_iv = ols(y, X_iv)

    # Correlaciones clave
    corr_p_ud = np.corrcoef(df["Precio"], df["Shock_demanda"])[0, 1]
    corr_costo_p = np.corrcoef(df["Costo"], df["Precio"])[0, 1]
    corr_costo_ud = np.corrcoef(df["Costo"], df["Shock_demanda"])[0, 1]

    # R2 primera etapa
    ssr_first = np.sum((precio - precio_hat) ** 2)
    sst_first = np.sum((precio - precio.mean()) ** 2)
    r2_first = 1 - ssr_first / sst_first

    # F parcial aproximado para instrumento costo condicionado en ingreso:
    # Se calcula como comparación entre modelo restringido Precio ~ Ingreso y completo Precio ~ Ingreso + Costo.
    X_restringido = np.column_stack([uno, ingreso])
    beta_r, _, _, precio_hat_r, _ = ols(precio, X_restringido)
    ssr_r = np.sum((precio - precio_hat_r) ** 2)
    ssr_u = ssr_first
    q = 1
    n = len(df)
    k_u = Z.shape[1]
    f_parcial = ((ssr_r - ssr_u) / q) / (ssr_u / (n - k_u))

    return {
        "beta_ols": beta_ols,
        "se_ols": se_ols,
        "t_ols": t_ols,
        "yhat_ols": yhat_ols,
        "resid_ols": resid_ols,
        "gamma": gamma,
        "se_gamma": se_gamma,
        "t_gamma": t_gamma,
        "precio_hat": precio_hat,
        "beta_iv": beta_iv,
        "se_iv": se_iv,
        "t_iv": t_iv,
        "yhat_iv": yhat_iv,
        "resid_iv": resid_iv,
        "corr_p_ud": corr_p_ud,
        "corr_costo_p": corr_costo_p,
        "corr_costo_ud": corr_costo_ud,
        "r2_first": r2_first,
        "f_parcial": f_parcial
    }


def tabla_coeficientes(beta, se, t, nombres):
    return pd.DataFrame({
        "Parámetro": nombres,
        "Coeficiente": beta,
        "Error estándar": se,
        "t calculado": t
    })


def explicar_signo_precio(coef):
    if coef < 0:
        return "El signo es negativo: al aumentar el precio, la cantidad demandada estimada disminuye. Esto coincide con la intuición económica de una curva de demanda."
    if coef > 0:
        return "El signo es positivo: esto sería sospechoso para una demanda, porque indicaría que mayor precio se asocia con mayor cantidad demandada. Puede ser efecto de simultaneidad o mala identificación."
    return "El coeficiente es cercano a cero: el modelo no estaría capturando una relación clara entre precio y cantidad."


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Panel de simulación")
st.sidebar.caption("Modifica los parámetros para mostrar cómo cambia el problema econométrico.")

n = st.sidebar.slider("Número de observaciones", 100, 2000, 500, 100)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=1, max_value=9999, value=42, step=1)
beta_precio_demanda = st.sidebar.slider("Coeficiente real del precio en demanda", -3.0, -0.2, -1.2, 0.1)
alpha_precio_oferta = st.sidebar.slider("Coeficiente real del precio en oferta", 0.2, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Objetivo de la app**

Mostrar que, cuando precio y cantidad se determinan al mismo tiempo, una regresión OLS puede estar sesgada.  
La solución presentada es estimar la ecuación estructural usando variables instrumentales / mínimos cuadrados en dos etapas.
""")

df = simular_oferta_demanda(
    n=n,
    seed=int(seed),
    beta_precio_demanda=beta_precio_demanda,
    alpha_precio_oferta=alpha_precio_oferta
)
res = estimar_modelos(df)


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>Ecuaciones estructurales en econometría</h1>
    <p><b>Presentado por:</b> Juan Pablo Vargas</p>
    <p>Modelo de ecuaciones simultáneas, identificación y estimación por variables instrumentales.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Portada",
    "2. Idea central",
    "3. Sistema estructural",
    "4. Simultaneidad",
    "5. Identificación",
    "6. Estimación 2SLS",
    "7. Resultados",
    "8. Referencias"
])


# ============================================================
# TAB 1
# ============================================================
with tab1:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("""
        <div class="card">
            <h2>¿De qué trata esta exposición?</h2>
            <p>
            En econometría, una <b>ecuación estructural</b> representa una relación económica de fondo.
            No busca solo describir correlaciones, sino representar una relación causal planteada por la teoría.
            </p>
            <p>
            El ejemplo central será un mercado donde <b>precio</b> y <b>cantidad</b> se determinan simultáneamente
            mediante una ecuación de demanda y una ecuación de oferta.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-card">
            <b>Claves:</b><br>
            El problema principal no es correr una regresión, sino saber si la ecuación que estoy estimando
            representa una relación causal identificable. Si una variable explicativa se determina dentro del
            sistema, OLS puede ser incorrecto.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>Ruta de la presentación</h3>
            <ol>
                <li>Qué es una ecuación estructural.</li>
                <li>Por qué aparece la simultaneidad.</li>
                <li>Por qué OLS puede fallar.</li>
                <li>Qué significa identificar una ecuación.</li>
                <li>Cómo se estima con 2SLS.</li>
                <li>Cómo interpretar los resultados.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Variable endógena central", "Precio")
        st.metric("Método recomendado", "2SLS / IV")


# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.markdown("""
    <div class="card">
        <h2>1. Idea central</h2>
        <p>
        Una regresión tradicional suele partir de una ecuación como:
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"Y = \beta_0 + \beta_1 X + u")

    st.markdown("""
    <div class="mini-card">
        Esta ecuación funciona bien con OLS cuando la variable explicativa X es exógena, es decir,
        cuando no está correlacionada con el error del modelo.
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"Cov(X,u)=0")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-box">
            <h3>Forma reducida</h3>
            <p>Describe una variable endógena usando variables externas al sistema.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-box">
            <h3>Forma estructural</h3>
            <p>Representa una relación económica planteada por teoría.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-box">
            <h3>Identificación</h3>
            <p>Permite recuperar los parámetros estructurales a partir de los datos.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-card">
        <b>Interpretación:</b><br>
        En ecuaciones estructurales, la pregunta no es solamente “¿cuánto se relacionan X e Y?”.
        La pregunta es: “¿puedo interpretar este coeficiente como un efecto económico real?”.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.markdown("""
    <div class="card">
        <h2>2. Sistema estructural: oferta y demanda</h2>
        <p>
        En un mercado, el precio no se elige de forma externa. El precio y la cantidad se determinan juntos.
        Por eso se necesita un sistema de ecuaciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="eq-box">
            <b>Ecuación de demanda</b><br><br>
            Q = β₀ + β₁P + β₂Ingreso + u<sub>d</sub>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="mini-card">
            <b>Interpretación:</b><br>
            β₁ mide cómo cambia la cantidad demandada cuando cambia el precio,
            manteniendo constante el ingreso.
            En teoría se espera β₁ &lt; 0.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="eq-box">
            <b>Ecuación de oferta</b><br><br>
            Q = α₀ + α₁P + α₂Costo + u<sub>s</sub>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="mini-card">
            <b>Interpretación:</b><br>
            α₁ mide cómo cambia la cantidad ofrecida cuando cambia el precio.
            En teoría se espera α₁ &gt; 0.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-card">
        <b>La clave:</b> en equilibrio, la cantidad demandada es igual a la cantidad ofrecida.
        Por eso observamos una sola cantidad y un solo precio, aunque detrás existan dos relaciones económicas.
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"Q_d = Q_s = Q")

    # Gráfico conceptual oferta-demanda
    p = np.linspace(10, 100, 200)
    qd = 160 - 1.1 * p
    qs = 20 + 1.0 * p

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qd, y=p, mode="lines", name="Demanda"))
    fig.add_trace(go.Scatter(x=qs, y=p, mode="lines", name="Oferta"))
    fig.update_layout(
        title="Representación conceptual: equilibrio de oferta y demanda",
        xaxis_title="Cantidad",
        yaxis_title="Precio",
        template="plotly_white",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="mini-card">
        <b>Cómo leer la gráfica:</b><br>
        La curva de demanda tiene pendiente negativa: a mayor precio, menor cantidad demandada.
        La curva de oferta tiene pendiente positiva: a mayor precio, mayor cantidad ofrecida.
        El punto donde se cruzan representa el equilibrio observado en el mercado.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 4
# ============================================================
with tab4:
    st.markdown("""
    <div class="card">
        <h2>3. Problema de simultaneidad</h2>
        <p>
        La simultaneidad ocurre cuando una variable explicativa se determina al mismo tiempo que la variable dependiente.
        En el ejemplo, el precio explica la cantidad, pero la cantidad de equilibrio también ayuda a determinar el precio.
        </p>
    </div>
    """, unsafe_allow_html=True)

    corr = res["corr_p_ud"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlación Precio - shock de demanda", f"{corr:,.3f}")
    c2.metric("Coeficiente real en demanda", f"{beta_precio_demanda:,.2f}")
    c3.metric("Coeficiente OLS estimado", f"{res['beta_ols'][1]:,.2f}")

    st.markdown("""
    <div class="danger-card">
        <b>Problema econométrico:</b><br>
        OLS necesita que la variable explicativa no esté correlacionada con el error.
        Pero si el precio se mueve por shocks de demanda no observados, el precio queda correlacionado con el error.
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"Cov(P,u_d)\neq 0")

    fig = px.scatter(
        df,
        x="Precio",
        y="Cantidad",
        trendline="ols",
        title="Relación observada entre precio y cantidad",
        labels={"Precio": "Precio observado", "Cantidad": "Cantidad observada"},
        template="plotly_white",
        opacity=0.72
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-card">
        <b>Interpretación:</b><br>
        La nube de puntos no muestra una curva pura de demanda. Muestra puntos de equilibrio generados por oferta y demanda.
        Por eso, una línea OLS puede mezclar ambas fuerzas y producir una estimación sesgada.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 5
# ============================================================
with tab5:
    st.markdown("""
    <div class="card">
        <h2>4. Identificación</h2>
        <p>
        Identificar una ecuación significa tener suficiente información para recuperar sus parámetros estructurales.
        En la práctica, se necesitan variables que muevan una ecuación, pero no entren directamente en la otra.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="success-card">
            <h3>Instrumento para la demanda</h3>
            <p>
            Para estimar la demanda, usamos <b>Costo</b> como instrumento del precio.
            </p>
            <p>
            El costo desplaza la oferta y afecta el precio de equilibrio,
            pero no debería afectar directamente la demanda del consumidor.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"Corr(Costo,Precio)\neq 0")
        st.latex(r"Corr(Costo,u_d)=0")

    with c2:
        st.markdown("""
        <div class="mini-card">
            <h3>Dos condiciones del instrumento</h3>
            <ol>
                <li><b>Relevancia:</b> el instrumento debe explicar la variable endógena.</li>
                <li><b>Exogeneidad:</b> el instrumento no debe estar correlacionado con el error estructural.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Corr(Costo, Precio)", f"{res['corr_costo_p']:,.3f}")
        st.metric("Corr(Costo, shock demanda)", f"{res['corr_costo_ud']:,.3f}")
        st.metric("F parcial primera etapa", f"{res['f_parcial']:,.2f}")

    # Gráfico instrumento vs precio
    fig = px.scatter(
        df,
        x="Costo",
        y="Precio",
        trendline="ols",
        title="Relevancia del instrumento: Costo explica Precio",
        labels={"Costo": "Costo de producción", "Precio": "Precio observado"},
        template="plotly_white",
        opacity=0.72
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-card">
        <b>Explicación:</b><br>
        Un instrumento sirve si mueve la variable problemática, pero no mueve directamente la variable dependiente,
        excepto a través de esa variable problemática.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 6
# ============================================================
with tab6:
    st.markdown("""
    <div class="card">
        <h2>5. Estimación por mínimos cuadrados en dos etapas</h2>
        <p>
        El método 2SLS reemplaza la parte endógena del precio por una parte predicha usando variables exógenas e instrumentos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="mini-card">
            <h3>Primera etapa</h3>
            <p>Se estima el precio usando variables exógenas e instrumentos.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"P = \pi_0 + \pi_1Ingreso + \pi_2Costo + v")
        st.markdown("""
        <div class="mini-card">
            El resultado de esta etapa es el precio estimado:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\hat{P}")

    with c2:
        st.markdown("""
        <div class="mini-card">
            <h3>Segunda etapa</h3>
            <p>Se estima la demanda usando el precio predicho, no el precio original.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"Q = \beta_0 + \beta_1\hat{P} + \beta_2Ingreso + u_d")

        st.markdown("""
        <div class="success-card">
            <b>Interpretación:</b><br>
            El precio predicho conserva la variación explicada por el instrumento,
            pero elimina la parte contaminada por el error de demanda.
        </div>
        """, unsafe_allow_html=True)

    comp = pd.DataFrame({
        "Precio observado": df["Precio"],
        "Precio predicho por primera etapa": res["precio_hat"]
    })

    fig = px.scatter(
        comp,
        x="Precio observado",
        y="Precio predicho por primera etapa",
        title="Primera etapa: comparación entre precio observado y precio predicho",
        template="plotly_white",
        opacity=0.72
    )
    fig.add_trace(go.Scatter(
        x=[comp["Precio observado"].min(), comp["Precio observado"].max()],
        y=[comp["Precio observado"].min(), comp["Precio observado"].max()],
        mode="lines",
        name="Referencia 45°"
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-card">
        <b>Interpretación:</b><br>
        Si los puntos siguen una relación positiva clara, la primera etapa está capturando parte importante del precio.
        Si no hubiera relación, el instrumento sería débil y la estimación 2SLS perdería confiabilidad.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 7
# ============================================================
with tab7:
    st.markdown("""
    <div class="card">
        <h2>6. Comparación de resultados</h2>
        <p>
        Aquí se compara el modelo OLS ingenuo contra el modelo 2SLS.
        El objetivo es mostrar por qué la estimación estructural requiere corregir la endogeneidad.
        </p>
    </div>
    """, unsafe_allow_html=True)

    nombres = ["Intercepto", "Precio", "Ingreso"]

    tabla_ols = tabla_coeficientes(res["beta_ols"], res["se_ols"], res["t_ols"], nombres)
    tabla_iv = tabla_coeficientes(res["beta_iv"], res["se_iv"], res["t_iv"], nombres)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("OLS ingenuo")
        st.dataframe(tabla_ols.style.format({
            "Coeficiente": "{:,.4f}",
            "Error estándar": "{:,.4f}",
            "t calculado": "{:,.2f}"
        }), use_container_width=True)

        st.markdown(f"""
        <div class="warning-card">
            <b>Lectura del coeficiente de precio en OLS:</b><br>
            El coeficiente estimado es <b>{res['beta_ols'][1]:,.3f}</b>.
            Este valor puede estar sesgado porque usa el precio observado, el cual se determina dentro del sistema.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("2SLS / Variables instrumentales")
        st.dataframe(tabla_iv.style.format({
            "Coeficiente": "{:,.4f}",
            "Error estándar": "{:,.4f}",
            "t calculado": "{:,.2f}"
        }), use_container_width=True)

        st.markdown(f"""
        <div class="success-card">
            <b>Lectura del coeficiente de precio en 2SLS:</b><br>
            El coeficiente estimado es <b>{res['beta_iv'][1]:,.3f}</b>.
            {explicar_signo_precio(res['beta_iv'][1])}
        </div>
        """, unsafe_allow_html=True)

    resumen = pd.DataFrame({
        "Modelo": ["Coeficiente real", "OLS ingenuo", "2SLS / IV"],
        "Coeficiente del precio en demanda": [
            beta_precio_demanda,
            res["beta_ols"][1],
            res["beta_iv"][1]
        ]
    })

    fig = px.bar(
        resumen,
        x="Modelo",
        y="Coeficiente del precio en demanda",
        text="Coeficiente del precio en demanda",
        title="Comparación del coeficiente del precio",
        template="plotly_white"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="mini-card">
        <b>Conclusión técnica:</b><br>
        Si el precio es endógeno, OLS no estima correctamente la demanda estructural.
        El método 2SLS usa un instrumento para aislar una variación del precio que sea útil para estimar el efecto causal.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 8
# ============================================================
with tab8:
    st.markdown("""
    <div class="card">
        <h2>8. Referencias bibliográficas</h2>
        <p>
        Las siguientes referencias sirven como base teórica para la exposición sobre ecuaciones estructurales,
        simultaneidad, identificación y estimación por variables instrumentales.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mini-card">
        <h3>Referencia principal</h3>
        <p>
        Wooldridge, J. M. (2010). <i>Introducción a la econometría: un enfoque moderno</i>.
        Cengage Learning.
        </p>
        <p>
        <b>Capítulo recomendado:</b> Capítulo 13, relacionado con modelos de ecuaciones simultáneas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mini-card">
        <h3>Referencias complementarias</h3>
        <ul>
            <li>
                Gujarati, D. N., & Porter, D. C. (2010).
                <i>Econometría</i>. McGraw-Hill.
            </li>
            <li>
                Greene, W. H. (2012).
                <i>Econometric Analysis</i>. Pearson.
            </li>
            <li>
                Stock, J. H., & Watson, M. W. (2012).
                <i>Introducción a la econometría</i>. Pearson.
            </li>
            <li>
                Hayashi, F. (2000).
                <i>Econometrics</i>. Princeton University Press.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-card">
        <h3>Cómo se usaron estas referencias</h3>
        <p>
        La exposición se apoya principalmente en la explicación econométrica de los modelos de ecuaciones simultáneas.
        En particular, se retoman los conceptos de:
        </p>
        <ul>
            <li>Ecuaciones estructurales.</li>
            <li>Variables endógenas y exógenas.</li>
            <li>Simultaneidad entre variables económicas.</li>
            <li>Problema de sesgo en mínimos cuadrados ordinarios.</li>
            <li>Identificación de ecuaciones.</li>
            <li>Variables instrumentales.</li>
            <li>Mínimos cuadrados en dos etapas, 2SLS.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)