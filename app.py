import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.special import gamma as gamma_func

# ==============================================================================
# 0. CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="Reliability Pro", page_icon="📈", layout="wide")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. DESIGN SYSTEM
# ==============================================================================

@dataclass(frozen=True)
class ProfessionalTheme:
    bg: str = '#FFFFFF'
    sidebar_bg: str = '#F8FAFC'
    text_main: str = '#0F172A'
    text_muted: str = '#64748B'
    grid: str = '#E2E8F0'
    colors: Tuple[str, ...] = ('#2563EB', '#4F46E5', '#059669', '#DC2626', '#D946EF')
    ci_alpha: float = 0.2  # Transparência do Intervalo de Confiança
    
THEME = ProfessionalTheme()

# ==============================================================================
# 2. GESTÃO DE DEPENDÊNCIAS
# ==============================================================================

class DependencyManager:
    @staticmethod
    def check_dependencies() -> Tuple[bool, bool]:
        rel_avail = False
        scipy_avail = False
        try:
            import reliability
            rel_avail = True
        except ImportError:
            pass
        try:
            import scipy
            scipy_avail = True
        except ImportError:
            pass
        return rel_avail, scipy_avail

RELIABILITY_AVAIL, SCIPY_AVAIL = DependencyManager.check_dependencies()

if RELIABILITY_AVAIL:
    from reliability.Fitters import (
        Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Normal_2P, Fit_Exponential_1P, Fit_Gamma_2P
    )
    from reliability.Distributions import (
        Weibull_Distribution, Lognormal_Distribution, Normal_Distribution, 
        Exponential_Distribution, Gamma_Distribution
    )
    from reliability.Nonparametric import KaplanMeier, NelsonAalen

if SCIPY_AVAIL:
    from scipy.stats import weibull_min, lognorm, norm, expon, gamma

# ==============================================================================
# 3. MOTOR ESTATÍSTICO (COM CÁLCULO DE IC)
# ==============================================================================

class ReliabilityEngine:
    """Motor robusto com cálculo de Intervalos de Confiança."""
    
    @staticmethod
    def process_timestamps(df: pd.DataFrame, time_col: str, status_col: str) -> pd.DataFrame:
        try:
            df_clean = df.copy()
            is_numeric = pd.api.types.is_numeric_dtype(df_clean[time_col])
            
            if not is_numeric:
                try:
                    df_clean[time_col] = pd.to_numeric(df_clean[time_col])
                    is_numeric = True
                except ValueError:
                    is_numeric = False

            if is_numeric:
                df_clean['Tempo'] = df_clean[time_col].astype('float32')
            else:
                try:
                    df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors='coerce')
                    df_clean = df_clean.dropna(subset=[time_col]).sort_values(time_col)
                    if df_clean.empty: return pd.DataFrame()
                    start_time = df_clean[time_col].min()
                    df_clean['Tempo'] = (df_clean[time_col] - start_time).dt.total_seconds() / 3600.0
                    df_clean['Tempo'] = df_clean['Tempo'].astype('float32')
                except Exception as e:
                    logger.error(f"Erro data: {e}")
                    return pd.DataFrame()

            df_clean['Falha'] = pd.to_numeric(df_clean[status_col], errors='coerce').fillna(0).astype('int8')
            df_clean['Falha'] = df_clean['Falha'].apply(lambda x: 1 if x >= 1 else 0)

            df_clean = df_clean.dropna(subset=['Tempo'])
            mask_invalid = df_clean['Tempo'] <= 0
            if mask_invalid.any():
                df_clean.loc[mask_invalid, 'Tempo'] = 0.01
            
            return df_clean[['Tempo', 'Falha']].copy()
        except Exception as e:
            logger.error(f"Erro processamento: {e}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_simulated_data(n: int = 500, distribution: str = "Weibull", **params) -> Tuple[pd.DataFrame, Dict]:
        if not SCIPY_AVAIL: return pd.DataFrame(), {}
        np.random.seed(42)
        p = {
            'beta': params.get('beta', 2.0), 'eta': params.get('eta', 1000.0),
            'mu': params.get('mu', 7.0), 'sigma': params.get('sigma', 0.5),
            'mu_norm': params.get('mu_norm', 1000.0), 'sigma_norm': params.get('sigma_norm', 200.0),
            'lambda': params.get('lambda_param', 0.001)
        }
        
        if distribution == "Weibull":
            vida = weibull_min.rvs(p['beta'], scale=p['eta'], size=n)
            limit = p['eta'] * 2
        elif distribution == "Lognormal":
            vida = lognorm.rvs(s=p['sigma'], scale=np.exp(p['mu']), size=n)
            limit = np.exp(p['mu']) * 3
        elif distribution == "Normal":
            vida = norm.rvs(loc=p['mu_norm'], scale=p['sigma_norm'], size=n)
            limit = p['mu_norm'] + 3*p['sigma_norm']
        elif distribution == "Exponential":
            vida = expon.rvs(scale=1/p['lambda'], size=n)
            limit = 3/p['lambda']
        else:
            vida = weibull_min.rvs(2.0, scale=1000, size=n)
            limit = 2000

        censura = np.random.uniform(limit*0.1, limit, n)
        tempo = np.minimum(vida, censura)
        status = (vida <= censura).astype(int)
        tempo = np.maximum(tempo, 0.01)
        
        return pd.DataFrame({'Tempo': tempo, 'Falha': status}), params

    @staticmethod
    def calculate_plot_limits(models: List[Dict], data_times: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if len(data_times) == 0: return (0, 100), (0, 1)
        x_max = data_times.max() * 1.2
        for m in models:
            try:
                dist = m['Distribuição']
                if hasattr(dist, 'quantile'):
                    q99 = dist.quantile(0.99)
                    if q99 < x_max * 5: x_max = max(x_max, q99)
            except: pass
        return (0, x_max), (0, 1.05)

    @staticmethod
    @st.cache_data(ttl=3600)
    def fit_models_cached(failures: np.ndarray, censored: np.ndarray) -> Optional[Tuple[pd.DataFrame, Dict]]:
        return ReliabilityEngine._fit_models(failures, censored)

    @staticmethod
    def _fit_models(failures: np.ndarray, censored: np.ndarray) -> Optional[Tuple[pd.DataFrame, Dict]]:
        if not RELIABILITY_AVAIL: return None
        failures = failures[failures > 0]
        censored = censored[censored > 0]
        if len(failures) < 2: return None
            
        results = []
        details = {}
        
        fitters = {
            'Weibull 2P': Fit_Weibull_2P, 'Lognormal 2P': Fit_Lognormal_2P,
            'Normal 2P': Fit_Normal_2P, 'Exponential 1P': Fit_Exponential_1P,
            'Gamma 2P': Fit_Gamma_2P
        }
        
        for name, FitterClass in fitters.items():
            try:
                # O parâmetro show_probability_plot=False evita gráficos ocultos
                fit = FitterClass(failures=failures, right_censored=censored, 
                                show_probability_plot=False, print_results=False)
                
                score = getattr(fit, 'AICc', getattr(fit, 'AIC', np.nan))
                if np.isnan(score) or np.isinf(score): continue

                params = {}
                for attr in ['alpha', 'beta', 'mu', 'sigma', 'Lambda', 'gamma']:
                    if hasattr(fit, attr): params[attr] = getattr(fit, attr)

                results.append({
                    'Modelo': name, 'AICc': score,
                    'Objeto': fit, 'Distribuição': fit.distribution
                })
                details[name] = {'params': params, 'fit_obj': fit}
            except Exception: continue
                
        if not results: return None
        df_res = pd.DataFrame(results).sort_values('AICc')
        df_res['ΔAICc'] = df_res['AICc'] - df_res['AICc'].min()
        return df_res, details

    @staticmethod
    def get_confidence_dists(fitter: Any, model_name: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Gera distribuições para os limites Inferior e Superior."""
        try:
            if 'Weibull' in model_name:
                # Limite Pessimista (Lower Alpha, Upper Beta) vs Otimista
                return (Weibull_Distribution(alpha=fitter.alpha_lower, beta=fitter.beta_upper),
                        Weibull_Distribution(alpha=fitter.alpha_upper, beta=fitter.beta_lower))
            elif 'Lognormal' in model_name:
                return (Lognormal_Distribution(mu=fitter.mu_lower, sigma=fitter.sigma_upper),
                        Lognormal_Distribution(mu=fitter.mu_upper, sigma=fitter.sigma_lower))
            elif 'Exponential' in model_name:
                return (Exponential_Distribution(Lambda=fitter.Lambda_upper),
                        Exponential_Distribution(Lambda=fitter.Lambda_lower))
            elif 'Normal' in model_name:
                return (Normal_Distribution(mu=fitter.mu_lower, sigma=fitter.sigma_upper),
                        Normal_Distribution(mu=fitter.mu_upper, sigma=fitter.sigma_lower))
            elif 'Gamma' in model_name:
                return (Gamma_Distribution(alpha=fitter.alpha_lower, beta=fitter.beta_upper),
                        Gamma_Distribution(alpha=fitter.alpha_upper, beta=fitter.beta_lower))
        except Exception: pass
        return None, None

# ==============================================================================
# 4. VISUALIZAÇÃO (COM ÁREA SOMBREADA DE IC)
# ==============================================================================

class ProfessionalVisualizer:
    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    @classmethod
    def plot_comparison(cls, models_data: List[Dict], t_plot: np.ndarray, 
                       emp_x: Optional[np.ndarray], emp_y: Optional[np.ndarray],
                       func_type: str, show_ci: bool) -> go.Figure:
        
        fig = go.Figure()
        
        # 1. Intervalos de Confiança (Desenhados primeiro para ficarem ao fundo)
        if show_ci and func_type in ['SF', 'CDF', 'CHF']:
            for i, m in enumerate(models_data):
                fitter = m.get('Objeto')
                if not fitter: continue
                
                # Obtém distribuições de limites
                dist_L, dist_U = ReliabilityEngine.get_confidence_dists(fitter, m['Modelo'])
                
                if dist_L and dist_U:
                    try:
                        y_l = getattr(dist_L, func_type)(t_plot)
                        y_u = getattr(dist_U, func_type)(t_plot)
                        
                        # Inverte ordem para fechar o polígono corretamente
                        x_poly = np.concatenate([t_plot, t_plot[::-1]])
                        y_poly = np.concatenate([y_u, y_l[::-1]])
                        
                        color_hex = THEME.colors[i % len(THEME.colors)]
                        color_rgba = cls.hex_to_rgba(color_hex, THEME.ci_alpha)
                        
                        fig.add_trace(go.Scatter(
                            x=x_poly, y=y_poly,
                            fill='toself',
                            fillcolor=color_rgba,
                            line=dict(width=0),
                            hoverinfo='skip',
                            showlegend=False,
                            name=f"IC {m['Modelo']}"
                        ))
                    except Exception: pass

        # 2. Dados Empíricos
        if emp_x is not None and emp_y is not None:
            is_step = func_type in ['SF', 'CHF', 'CDF']
            fig.add_trace(go.Scatter(
                x=emp_x, y=emp_y,
                mode='lines' if is_step else 'markers',
                line=dict(shape='hv', dash='dot', color='black', width=2) if is_step else None,
                marker=dict(color='black', opacity=0.6, size=5) if not is_step else None,
                name='Dados Empíricos'
            ))
            
        # 3. Curvas dos Modelos
        for i, m in enumerate(models_data):
            color = THEME.colors[i % len(THEME.colors)]
            try:
                y_val = getattr(m['Distribuição'], func_type)(t_plot)
                fig.add_trace(go.Scatter(
                    x=t_plot, y=y_val, mode='lines', name=m['Modelo'],
                    line=dict(color=color, width=2.5)
                ))
            except: continue
                
        fig.update_layout(
            template='plotly_white',
            title=dict(text=f"Análise: {func_type}", font=dict(size=18, color=THEME.text_main)),
            xaxis=dict(title="Tempo (h)", gridcolor=THEME.grid),
            yaxis=dict(title="Probabilidade / Valor", gridcolor=THEME.grid),
            legend=dict(orientation="h", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500
        )
        return fig

# ==============================================================================
# 5. APLICAÇÃO PRINCIPAL
# ==============================================================================

class ReliabilityApp:
    def __init__(self):
        self.apply_styles()
        self.engine = ReliabilityEngine()
        if 'df_processed' not in st.session_state: st.session_state.df_processed = None
        if 'ranking' not in st.session_state: st.session_state.ranking = None

    def apply_styles(self):
        st.markdown(f"""
            <style>
                .stApp {{ background-color: {THEME.bg}; }}
                div[data-testid="stMetricValue"] {{ font-size: 24px; color: {THEME.text_main}; }}
                .stDataFrame {{ font-size: 12px; }}
            </style>
        """, unsafe_allow_html=True)

    def sidebar_handler(self):
        with st.sidebar:
            st.title("🔧 Configuração")
            if st.button("🗑️ Limpar Tudo", type="secondary"):
                st.session_state.clear()
                st.rerun()
            st.divider()
            
            mode = st.radio("Fonte de Dados", ["Upload CSV", "Simulação"])
            
            if mode == "Upload CSV":
                file = st.file_uploader("Arquivo CSV", type=["csv"])
                if file:
                    try:
                        df_raw = pd.read_csv(file)
                        st.success(f"Lido: {df_raw.shape[0]} linhas")
                        c1, c2 = st.columns(2)
                        t_col = c1.selectbox("Tempo/Data", df_raw.columns)
                        f_col = c2.selectbox("Falha (0/1)", df_raw.columns)
                        
                        if st.button("Processar", type="primary"):
                            df_proc = self.engine.process_timestamps(df_raw, t_col, f_col)
                            if not df_proc.empty:
                                st.session_state.df_processed = df_proc
                                st.session_state.ranking = None
                                st.success("Processado!")
                                st.rerun()
                            else: st.error("Erro: Dados inválidos.")
                    except Exception as e: st.error(f"Erro CSV: {e}")
                        
            else: # Simulação
                dist = st.selectbox("Distribuição", ["Weibull", "Lognormal", "Normal", "Exponential"])
                n = st.slider("Amostras", 100, 5000, 1000)
                params = {}
                if dist == "Weibull":
                    params['beta'] = st.slider("Beta", 0.5, 5.0, 2.0)
                    params['eta'] = st.slider("Eta", 500, 5000, 1500)
                
                if st.button("Gerar Dados"):
                    df_sim, _ = self.engine.generate_simulated_data(n, dist, **params)
                    st.session_state.df_processed = df_sim
                    st.session_state.ranking = None
                    st.success(f"Simulação ({dist}) OK!")
                    st.rerun()

    def run_analysis(self):
        if st.session_state.df_processed is None: return
        df = st.session_state.df_processed
        
        if st.session_state.ranking is None:
            with st.spinner("Calculando modelos..."):
                falhas = df[df['Falha'] == 1]['Tempo'].values
                censuras = df[df['Falha'] == 0]['Tempo'].values
                
                result = self.engine.fit_models_cached(falhas, censuras)
                if result:
                    st.session_state.ranking, st.session_state.model_details = result
                    if RELIABILITY_AVAIL:
                        try:
                            km = KaplanMeier(failures=falhas, right_censored=censuras, show_plot=False, print_results=False)
                            na = NelsonAalen(failures=falhas, right_censored=censuras, show_plot=False, print_results=False)
                            st.session_state.emp_data = {
                                'x_km': km.xvals, 'y_km': km.SF, 'x_na': na.xvals, 'y_na': na.CHF
                            }
                        except: st.session_state.emp_data = {}
                else: st.error("Dados insuficientes para ajuste.")

    def main_view(self):
        if st.session_state.df_processed is None:
            st.info("👈 Carregue dados na barra lateral.")
            return

        df = st.session_state.df_processed
        ranking = st.session_state.get('ranking')

        # KPI
        n_total = len(df)
        n_falhas = df['Falha'].sum()
        best = ranking.iloc[0]['Modelo'] if ranking is not None else "..."
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", n_total)
        c2.metric("Falhas", n_falhas)
        c3.metric("Censuras", n_total - n_falhas)
        c4.metric("Melhor Ajuste", best)

        if ranking is None: return

        st.divider()
        
        # Plot Controls
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            models = ranking['Modelo'].tolist()
            sel_models = st.multiselect("Comparar", models, default=[models[0]])
        with c2:
            func = st.selectbox("Função", ["SF", "CDF", "PDF", "HF", "CHF"])
        with c3:
            # CHECKBOX NOVO PARA LIGAR/DESLIGAR IC
            show_ci = st.checkbox("Mostrar IC 95%", value=True)

        # Prepare Plot Data
        models_to_plot = []
        if st.session_state.get('model_details'):
            for m in sel_models:
                row = ranking[ranking['Modelo'] == m].iloc[0]
                models_to_plot.append({
                    'Modelo': m, 
                    'Distribuição': row['Objeto'].distribution,
                    'Objeto': row['Objeto'] # Necessário para pegar os limites
                })

        emp_data = st.session_state.get('emp_data', {})
        x_emp, y_emp = None, None
        if func == "SF": x_emp, y_emp = emp_data.get('x_km'), emp_data.get('y_km')
        elif func == "CDF" and emp_data.get('y_km') is not None: x_emp, y_emp = emp_data.get('x_km'), 1 - emp_data.get('y_km')
        elif func == "CHF": x_emp, y_emp = emp_data.get('x_na'), emp_data.get('y_na')

        (xmin, xmax), _ = self.engine.calculate_plot_limits(models_to_plot, df['Tempo'].values)
        t_plot = np.linspace(xmin, xmax, 200)

        # Chama visualizador passando show_ci
        fig = ProfessionalVisualizer.plot_comparison(models_to_plot, t_plot, x_emp, y_emp, func, show_ci)
        st.plotly_chart(fig, use_container_width=True)

        # Tabs
        t1, t2, t3 = st.tabs(["🏆 Ranking", "⚙️ Parâmetros", "📋 Dados"])
        with t1:
            st.dataframe(ranking[['Modelo', 'AICc', 'ΔAICc']].style.highlight_min(subset=['ΔAICc'], color='#e0f2fe'), use_container_width=True)
        with t2:
            if st.session_state.get('model_details'):
                p_list = []
                for m in models:
                    d = st.session_state.model_details[m]['params']
                    flat = {k: f"{v:.4f}" for k, v in d.items()}
                    flat['Modelo'] = m
                    p_list.append(flat)
                st.dataframe(pd.DataFrame(p_list).set_index('Modelo'), use_container_width=True)
        with t3:
            st.dataframe(df, use_container_width=True)

    def run(self):
        self.sidebar_handler()
        self.run_analysis()
        self.main_view()

if __name__ == "__main__":
    app = ReliabilityApp()
    app.run()