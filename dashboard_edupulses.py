import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
import os
import numpy as np
from google.cloud import storage

# --- SEU CÓDIGO ORIGINAL - FUNÇÕES E CONSTANTES (PRESERVADO 100%) ---

# Configuração da página Streamlit
st.set_page_config(layout="wide", page_title="Edupulses B2B Dashboard")

# --- Constantes e Definições de Padrões ---
NOME_ARQUIVO_MAPEAMENTO = "mapeamento_descricoes_edupulses.csv"
PLANOS_KEYWORDS = {'Pop': r'pop', 'Profissional': r'profissional', 'Premium': r'premium'}
PLANOS_VALIDOS = ['Pop', 'Profissional', 'Premium', 'Assincrono', 'Outros']
PLANOS_DROPDOWN_OPCOES = ["Sem plano definido"] + PLANOS_VALIDOS
PERIODICIDADE_KEYWORDS = {'Anual': r'anual', 'Semestral': r'semestral', 'Mensal': r'mensal'}
PERIODICIDADE_VALIDAS = ['Anual', 'Semestral', 'Mensal']
PERIODICIDADE_DROPDOWN_OPCOES = PERIODICIDADE_VALIDAS
STATUS_CLIENTE_VALIDOS = ['Ativo', 'Inativo (Churned)']
GCP_BUCKET_NAME = "dash-edu050823"
GCP_FILE_PATH = "mapeamento_descricoes_edupulses.csv" # Nome do arquivo dentro do bucket
APP_USERNAME_ENV_VAR = "APP_USERNAME"
APP_PASSWORD_ENV_VAR = "APP_PASSWORD"

def check_credentials(username_input, password_input):
    """Verifica as credenciais contra as variáveis de ambiente."""
    correct_username = os.environ.get(APP_USERNAME_ENV_VAR)
    correct_password = os.environ.get(APP_PASSWORD_ENV_VAR)

    # Para teste local, se as variáveis de ambiente não estiverem definidas:
    if correct_username is None or correct_password is None:
        st.warning("Credenciais de admin não configuradas no ambiente. Usando defaults para teste.")
        correct_username = correct_username or "Edupulses"  # Default para teste local
        correct_password = correct_password or "7817jbxjx@lshs987" # Default para teste local
        # NÃO USE ESSES DEFAULTS EM PRODUÇÃO. Configure as variáveis de ambiente no Cloud Run.

    if username_input == correct_username and password_input == correct_password:
        return True
    return False

def login_form():
    """Exibe o formulário de login e gerencia o estado de autenticação."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.header("Login - Dashboard Edupulses B2B")
        username = st.text_input("Usuário", key="login_user")
        password = st.text_input("Senha", type="password", key="login_pass")

        if st.button("Entrar", key="login_button"):
            if check_credentials(username, password):
                st.session_state.authenticated = True
                st.rerun() # Re-executa para remover o formulário e mostrar o dashboard
            else:
                st.error("Usuário ou senha incorretos.")
        return False # Indica que o usuário não está autenticado
    return True # Indica que o usuário está autenticado

def logout():
    """Realiza o logout do usuário."""
    st.session_state.authenticated = False
    st.rerun() # Re-executa para mostrar o formulário de login novamente

# --- Funções de Mapeamento ---
def carregar_mapeamentos_salvos(GCP_BUCKET_NAME, GCP_FILE_PATH):
    mapeamentos = {}
    gcs_uri = f"gs://{GCP_BUCKET_NAME}/{GCP_FILE_PATH}"
    try:
        # Usa o gcsfs (que o pandas usa por baixo dos panos) para ler o CSV
        # diretamente do bucket para o pandas.
        df_map = pd.read_csv(gcs_uri)

        for _, row in df_map.iterrows():
            # Garante que as colunas esperadas existam antes de tentar acessá-las
            plano_corrigido_val = row.get('plano_corrigido')
            periodicidade_corrigida_val = row.get('periodicidade_corrigida')
            descricao_original_val = row.get('descricao_original')

            plano_corrigido = plano_corrigido_val if pd.notna(plano_corrigido_val) else "Sem plano definido"
            periodicidade_corrigida = periodicidade_corrigida_val if pd.notna(periodicidade_corrigida_val) else "Mensal"

            if periodicidade_corrigida not in PERIODICIDADE_VALIDAS:
                periodicidade_corrigida = "Mensal"

            if pd.notna(descricao_original_val):
                mapeamentos[descricao_original_val] = {'plano': plano_corrigido, 'periodicidade': periodicidade_corrigida}
            else:
                # Log ou aviso caso a coluna essencial 'descricao_original' esteja faltando em alguma linha
                # st.sidebar.warning(f"Linha no arquivo de mapeamento sem 'descricao_original': {row.to_dict()}")
                pass # Ou pode decidir pular esta linha

        if not mapeamentos: # Se o arquivo foi lido mas estava vazio ou sem descrições válidas
             st.sidebar.info(f"Arquivo de mapeamento '{GCP_FILE_PATH}' lido do Cloud, mas está vazio ou sem descrições válidas.")
        else:
            st.sidebar.success(f"Mapeamentos carregados de '{GCP_FILE_PATH}' no Cloud Storage!")

    except FileNotFoundError: # Especificamente para quando o gcsfs não encontra o arquivo
        st.sidebar.info(f"Arquivo de mapeamento '{GCP_FILE_PATH}' não encontrado no Cloud Storage. Será criado ao salvar.")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar mapeamentos do Cloud: {e}")
        st.sidebar.info("Um arquivo de mapeamento vazio será usado.")
    return mapeamentos

def aplicar_mapeamentos_ao_df(df, mapeamentos):
    if not mapeamentos or df is None: return df
    df_copia = df.copy()
    if 'descricao_servico' not in df_copia.columns:
        return df_copia
    for desc_servico, correcao in mapeamentos.items():
        mask_correcao = df_copia['descricao_servico'] == desc_servico
        if mask_correcao.any():
            if 'plano' in correcao and correcao['plano'] is not None:
                df_copia.loc[mask_correcao, 'nome_plano_extraido'] = correcao['plano']
                df_copia.loc[mask_correcao, 'plano_identificado'] = correcao['plano'] not in ["Sem plano definido", None]
            if 'periodicidade' in correcao and correcao['periodicidade'] is not None:
                if correcao['periodicidade'] in PERIODICIDADE_VALIDAS:
                    df_copia.loc[mask_correcao, 'periodicidade_extraida'] = correcao['periodicidade']
                    df_copia.loc[mask_correcao, 'periodicidade_identificada'] = True
                else:
                    df_copia.loc[mask_correcao, 'periodicidade_extraida'] = "Mensal"
                    df_copia.loc[mask_correcao, 'periodicidade_identificada'] = True
    return df_copia

# --- Funções de Processamento de Dados ---
def carregar_dados(arquivo_upload):
    if arquivo_upload is not None:
        try:
            file_name = arquivo_upload.name
            if file_name.endswith('.csv'):
                df = pd.read_csv(arquivo_upload)
                st.success(f"Arquivo CSV '{file_name}' carregado com sucesso!")
                return df
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(arquivo_upload, engine='openpyxl')
                st.success(f"Arquivo Excel '{file_name}' carregado com sucesso!")
                return df
            else:
                st.error("Formato de arquivo não suportado. Por favor, envie .csv ou .xlsx")
                return None
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo '{arquivo_upload.name}': {e}")
            return None
    return None

def limpar_valor_monetario(valor):
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        valor_limpo = str(valor).replace('R$', '').replace('.', '').replace(',', '.').strip()
        try:
            return float(valor_limpo)
        except ValueError:
            return None
    return None

def processar_dados_iniciais(df_raw):
    if df_raw is None: return None
    df = df_raw.copy()
    colunas_para_manter = {
        'Tipo do cliente': 'tipo_cliente', 'Nome do cliente': 'nome_cliente',
        'Documento do cliente': 'documento_cliente', 'Data de emissão': 'data_emissao',
        'Descrição do serviço': 'descricao_servico', 'Valor total': 'valor_total_original'
    }
    colunas_existentes = {k: v for k, v in colunas_para_manter.items() if k in df.columns}
    colunas_essenciais_req = ['Descrição do serviço', 'Data de emissão', 'Documento do cliente', 'Valor total']
    colunas_faltantes_req = [col for col in colunas_essenciais_req if col not in colunas_existentes.keys()]
    if colunas_faltantes_req:
        st.error(f"Colunas essenciais não encontradas no arquivo: {', '.join(colunas_faltantes_req)}. Verifique o arquivo.")
        st.info(f"Colunas encontradas no arquivo: {df.columns.tolist()}")
        return None
    df = df[list(colunas_existentes.keys())].rename(columns=colunas_existentes)
    df['data_emissao'] = pd.to_datetime(df['data_emissao'], errors='coerce')
    df.dropna(subset=['data_emissao'], inplace=True)
    df['valor_total'] = df['valor_total_original'].apply(limpar_valor_monetario)
    df['documento_cliente'] = df['documento_cliente'].astype(str).str.strip().str.replace(r'[^0-9]', '', regex=True)
    df.dropna(subset=['documento_cliente'], inplace=True)
    df = df[df['documento_cliente'] != '']
    if 'tipo_cliente' not in df.columns:
        df['tipo_cliente'] = "Não Especificado"
    else:
        df['tipo_cliente'] = df['tipo_cliente'].astype(str).fillna("Não Especificado")
    df['descricao_servico_lower'] = df['descricao_servico'].astype(str).str.lower()
    df['nome_plano_extraido'] = "Sem plano definido"
    df['plano_identificado'] = False
    for nome_plano, keyword_regex in PLANOS_KEYWORDS.items():
        mask = df['descricao_servico_lower'].str.contains(keyword_regex, regex=True, na=False)
        df.loc[mask, 'nome_plano_extraido'] = nome_plano
        df.loc[mask, 'plano_identificado'] = True
    df['periodicidade_extraida'] = "Sem periodicidade definida"
    df['periodicidade_identificada'] = False
    for nome_periodicidade, keyword_regex in PERIODICIDADE_KEYWORDS.items():
        mask_period = df['descricao_servico_lower'].str.contains(keyword_regex, regex=True, na=False)
        df.loc[mask_period, 'periodicidade_extraida'] = nome_periodicidade
        df.loc[mask_period, 'periodicidade_identificada'] = True
    df.drop(columns=['descricao_servico_lower'], inplace=True, errors='ignore')
    if 'nome_cliente' not in df.columns: df['nome_cliente'] = "Cliente não informado"
    else: df['nome_cliente'] = df['nome_cliente'].astype(str).fillna("Cliente não informado")
    return df

def calcular_metricas_fatura(df):
    if df is None: return None
    df_calculado = df.copy()
    df_calculado['duracao_dias_plano'] = 0
    df_calculado['valor_mensal_plano'] = None
    df_calculado['data_fim_estimada_fatura'] = pd.NaT
    anual_mask = df_calculado['periodicidade_extraida'] == 'Anual'
    df_calculado.loc[anual_mask, 'duracao_dias_plano'] = 365
    df_calculado.loc[anual_mask, 'valor_mensal_plano'] = df_calculado.loc[anual_mask, 'valor_total'] / 12
    semestral_mask = df_calculado['periodicidade_extraida'] == 'Semestral'
    df_calculado.loc[semestral_mask, 'duracao_dias_plano'] = 180
    df_calculado.loc[semestral_mask, 'valor_mensal_plano'] = df_calculado.loc[semestral_mask, 'valor_total'] / 6
    mensal_mask = df_calculado['periodicidade_extraida'] == 'Mensal'
    df_calculado.loc[mensal_mask, 'duracao_dias_plano'] = 30
    df_calculado.loc[mensal_mask, 'valor_mensal_plano'] = df_calculado.loc[mensal_mask, 'valor_total'] / 1
    default_mensal_mask = ~df_calculado['periodicidade_extraida'].isin(PERIODICIDADE_VALIDAS)
    df_calculado.loc[default_mensal_mask, 'periodicidade_extraida'] = 'Mensal'
    df_calculado.loc[default_mensal_mask, 'periodicidade_identificada'] = True
    df_calculado.loc[default_mensal_mask, 'duracao_dias_plano'] = 30
    df_calculado.loc[default_mensal_mask, 'valor_mensal_plano'] = df_calculado.loc[default_mensal_mask, 'valor_total'] / 1
    df_calculado['valor_mensal_plano'] = np.where(pd.isna(df_calculado['valor_total']), 0, df_calculado['valor_mensal_plano'])
    df_calculado['valor_mensal_plano'].fillna(0, inplace=True)
    valid_duration_mask = pd.notna(df_calculado['data_emissao']) & (df_calculado['duracao_dias_plano'] > 0)
    df_calculado.loc[valid_duration_mask, 'data_fim_estimada_fatura'] = df_calculado.loc[valid_duration_mask, 'data_emissao'] + pd.to_timedelta(df_calculado.loc[valid_duration_mask, 'duracao_dias_plano'], unit='D')
    df_calculado.loc[pd.isna(df_calculado['data_fim_estimada_fatura']) & pd.notna(df_calculado['data_emissao']), 'data_fim_estimada_fatura'] = \
        df_calculado.loc[pd.isna(df_calculado['data_fim_estimada_fatura']) & pd.notna(df_calculado['data_emissao']), 'data_emissao'] + pd.to_timedelta(30, unit='D')
    return df_calculado

def adicionar_metricas_cliente(df_faturas):
    if df_faturas is None or 'documento_cliente' not in df_faturas.columns: return df_faturas
    df_clientes_agg = df_faturas.groupby('documento_cliente').agg(
        data_entrada_cliente=('data_emissao', 'min'),
        data_ultima_fatura_cliente=('data_emissao', 'max'),
        total_gasto_cliente=('valor_total', 'sum'),
        numero_faturas_cliente=('documento_cliente', 'count')
    ).reset_index()
    df_faturas_enriquecido = pd.merge(df_faturas, df_clientes_agg, on='documento_cliente', how='left')
    return df_faturas_enriquecido

def analisar_ciclo_de_vida_clientes(df_faturas_enriquecido, data_referencia):
    if df_faturas_enriquecido is None or 'documento_cliente' not in df_faturas_enriquecido.columns:
        st.warning("DataFrame de faturas não encontrado para análise de ciclo de vida.")
        return df_faturas_enriquecido
    df = df_faturas_enriquecido.sort_values(by=['documento_cliente', 'data_emissao']).copy()
    df['data_fim_estimada_fatura'] = pd.to_datetime(df['data_fim_estimada_fatura'], errors='coerce')
    df['prazo_renovacao_fatura'] = pd.NaT
    df['data_churn_cliente'] = pd.NaT
    df['status_cliente'] = 'Pendente'
    clientes_unicos = df['documento_cliente'].unique()
    for cliente_doc in clientes_unicos:
        faturas_cliente = df[df['documento_cliente'] == cliente_doc].sort_values(by='data_emissao')
        is_churned_cliente_final = False
        data_churn_efetiva_final = pd.NaT
        if faturas_cliente.empty: continue
        ultima_fatura_do_cliente = faturas_cliente.iloc[-1]
        data_fim_ultima_fatura = ultima_fatura_do_cliente['data_fim_estimada_fatura']
        df.loc[faturas_cliente.index, 'prazo_renovacao_fatura'] = pd.to_datetime(faturas_cliente['data_fim_estimada_fatura'], errors='coerce') + timedelta(days=90)
        if pd.isna(data_fim_ultima_fatura):
            if ultima_fatura_do_cliente['data_emissao'] < data_referencia - timedelta(days=120):
                is_churned_cliente_final = True
                data_churn_efetiva_final = ultima_fatura_do_cliente['data_emissao']
            else:
                is_churned_cliente_final = False
        else:
            prazo_renovacao_ultima_fatura = data_fim_ultima_fatura + timedelta(days=90)
            faturas_apos_fim_ultima = faturas_cliente[
                (faturas_cliente['data_emissao'] > data_fim_ultima_fatura) &
                (faturas_cliente['data_emissao'] <= prazo_renovacao_ultima_fatura)
            ]
            renovacao_efetiva_encontrada = not faturas_apos_fim_ultima.empty
            if renovacao_efetiva_encontrada:
                is_churned_cliente_final = False
            else:
                if data_referencia > prazo_renovacao_ultima_fatura:
                    is_churned_cliente_final = True
                    data_churn_efetiva_final = data_fim_ultima_fatura
                elif data_fim_ultima_fatura < data_referencia <= prazo_renovacao_ultima_fatura:
                    is_churned_cliente_final = True
                    data_churn_efetiva_final = data_fim_ultima_fatura
                elif data_fim_ultima_fatura >= data_referencia:
                    is_churned_cliente_final = False
                else:
                    is_churned_cliente_final = False
        if is_churned_cliente_final:
            df.loc[df['documento_cliente'] == cliente_doc, 'data_churn_cliente'] = data_churn_efetiva_final
            df.loc[df['documento_cliente'] == cliente_doc, 'status_cliente'] = 'Inativo (Churned)'
        else:
            df.loc[df['documento_cliente'] == cliente_doc, 'status_cliente'] = 'Ativo'
            df.loc[df['documento_cliente'] == cliente_doc, 'data_churn_cliente'] = pd.NaT
    return df

def calcular_mrr_arpa_mensal(df_faturas_processado):
    if df_faturas_processado is None or df_faturas_processado.empty:
        return pd.DataFrame(columns=['MesAno', 'MRR_Total', 'Clientes_Ativos_Count', 'ARPA', 'MesAnoStr'])
    df_faturas_processado['data_emissao'] = pd.to_datetime(df_faturas_processado['data_emissao'])
    df_faturas_processado['data_fim_estimada_fatura'] = pd.to_datetime(df_faturas_processado['data_fim_estimada_fatura'])
    df_mrr_calc = df_faturas_processado[
        pd.notna(df_faturas_processado['valor_mensal_plano']) & (df_faturas_processado['valor_mensal_plano'] > 0) &
        pd.notna(df_faturas_processado['data_emissao']) &
        pd.notna(df_faturas_processado['data_fim_estimada_fatura'])
    ].copy()
    if df_mrr_calc.empty:
        return pd.DataFrame(columns=['MesAno', 'MRR_Total', 'Clientes_Ativos_Count', 'ARPA', 'MesAnoStr'])
    min_data = df_mrr_calc['data_emissao'].min()
    max_data_projecao = df_mrr_calc['data_fim_estimada_fatura'].max()
    if pd.isna(min_data) or pd.isna(max_data_projecao):
         return pd.DataFrame(columns=['MesAno', 'MRR_Total', 'Clientes_Ativos_Count', 'ARPA', 'MesAnoStr'])
    meses_analise = pd.date_range(min_data.replace(day=1), max_data_projecao.replace(day=1), freq='MS')
    if meses_analise.empty and pd.notna(min_data):
        meses_analise = pd.date_range(min_data.replace(day=1), min_data.replace(day=1), freq='MS')
    mrr_mensal_data = []
    for mes_inicio in meses_analise:
        mes_fim = mes_inicio + pd.offsets.MonthEnd(0)
        faturas_ativas_no_mes = df_mrr_calc[
            (df_mrr_calc['data_emissao'] <= mes_fim) &
            (df_mrr_calc['data_fim_estimada_fatura'] >= mes_inicio)
        ]
        mrr_total_mes = faturas_ativas_no_mes['valor_mensal_plano'].sum()
        clientes_ativos_count = faturas_ativas_no_mes['documento_cliente'].nunique()
        arpa_mes = (mrr_total_mes / clientes_ativos_count) if clientes_ativos_count > 0 else 0
        mrr_mensal_data.append({
            'MesAno': mes_inicio,
            'MRR_Total': mrr_total_mes,
            'Clientes_Ativos_Count': clientes_ativos_count,
            'ARPA': arpa_mes
        })
    df_mrr = pd.DataFrame(mrr_mensal_data)
    if not df_mrr.empty:
        df_mrr['MesAnoStr'] = df_mrr['MesAno'].dt.strftime('%Y-%m')
    else:
        df_mrr = pd.DataFrame(columns=['MesAno', 'MRR_Total', 'Clientes_Ativos_Count', 'ARPA', 'MesAnoStr'])
    return df_mrr

def calcular_metricas_permanencia(df_com_status_cliente):
    if df_com_status_cliente is None or df_com_status_cliente.empty:
        return pd.DataFrame(columns=['documento_cliente', 'data_entrada_cliente', 'status_cliente',
                                     'data_fim_relacionamento', 'permanencia_dias', 'permanencia_meses'])
    df = df_com_status_cliente.copy()
    df['data_entrada_cliente'] = pd.to_datetime(df['data_entrada_cliente'], errors='coerce')
    df['data_churn_cliente'] = pd.to_datetime(df['data_churn_cliente'], errors='coerce')
    df['data_fim_estimada_fatura'] = pd.to_datetime(df['data_fim_estimada_fatura'], errors='coerce')
    clientes_info = []
    df_clientes_unicos = df.drop_duplicates(subset=['documento_cliente'])
    for _, row_cliente in df_clientes_unicos.iterrows():
        doc_cliente = row_cliente['documento_cliente']
        data_entrada = row_cliente['data_entrada_cliente']
        status_atual = row_cliente['status_cliente']
        data_churn = row_cliente['data_churn_cliente']
        data_fim_relacionamento = pd.NaT
        if status_atual == 'Ativo':
            faturas_do_cliente_ativo = df[df['documento_cliente'] == doc_cliente]
            if not faturas_do_cliente_ativo.empty:
                 data_fim_relacionamento = faturas_do_cliente_ativo['data_fim_estimada_fatura'].max()
        elif status_atual == 'Inativo (Churned)':
            data_fim_relacionamento = data_churn
        permanencia_dias = np.nan
        permanencia_meses = np.nan
        if pd.notna(data_entrada) and pd.notna(data_fim_relacionamento) and data_fim_relacionamento >= data_entrada:
            permanencia_dias = (data_fim_relacionamento - data_entrada).days
            permanencia_meses = permanencia_dias / 30.4375
        clientes_info.append({
            'documento_cliente': doc_cliente,
            'data_entrada_cliente': data_entrada,
            'status_cliente': status_atual,
            'data_churn_cliente': data_churn,
            'data_fim_relacionamento': data_fim_relacionamento,
            'permanencia_dias': permanencia_dias,
            'permanencia_meses': permanencia_meses
        })
    df_permanencia_clientes = pd.DataFrame(clientes_info)
    return df_permanencia_clientes

def calcular_metricas_churn_mensal(df_clientes_unicos_com_status, data_final_base, df_mrr_mensal_para_ativos):
    if df_clientes_unicos_com_status is None or df_clientes_unicos_com_status.empty:
        return pd.DataFrame(columns=['MesAno', 'Churn_Count', 'Ativos_Inicio_Mes_Estimado', 'Taxa_Churn_Mensal', 'MesAnoStr'])
    df_churned = df_clientes_unicos_com_status[
        (df_clientes_unicos_com_status['status_cliente'] == 'Inativo (Churned)') &
        pd.notna(df_clientes_unicos_com_status['data_churn_cliente'])
    ].copy()
    df_metricas_churn = pd.DataFrame()
    min_data_geral = pd.NaT
    if not df_clientes_unicos_com_status['data_entrada_cliente'].dropna().empty:
        min_data_geral = df_clientes_unicos_com_status['data_entrada_cliente'].dropna().min()
    if pd.isna(min_data_geral) and not df_churned.empty and not pd.to_datetime(df_churned['data_churn_cliente'], errors='coerce').dropna().empty:
        min_data_geral = pd.to_datetime(df_churned['data_churn_cliente'], errors='coerce').dropna().min()
    if pd.isna(min_data_geral) or pd.isna(data_final_base):
        return pd.DataFrame(columns=['MesAno', 'Churn_Count', 'Ativos_Inicio_Mes_Estimado', 'Taxa_Churn_Mensal', 'MesAnoStr'])
    todos_os_meses_range = pd.date_range(start=min_data_geral.replace(day=1),
                                         end=data_final_base.replace(day=1),
                                         freq='MS')
    if todos_os_meses_range.empty and pd.notna(min_data_geral):
         todos_os_meses_range = pd.date_range(start=min_data_geral.replace(day=1),
                                              end=min_data_geral.replace(day=1),
                                              freq='MS')
    if todos_os_meses_range.empty:
        return pd.DataFrame(columns=['MesAno', 'Churn_Count', 'Ativos_Inicio_Mes_Estimado', 'Taxa_Churn_Mensal', 'MesAnoStr'])
    df_metricas_churn = pd.DataFrame({'MesAno': todos_os_meses_range})
    if not df_churned.empty:
        df_churned['data_churn_cliente'] = pd.to_datetime(df_churned['data_churn_cliente'])
        df_churned['MesAnoChurnPeriod'] = df_churned['data_churn_cliente'].dt.to_period('M')
        churn_mensal_agg = df_churned.groupby('MesAnoChurnPeriod').size().reset_index(name='Churn_Count')
        churn_mensal_agg['MesAno'] = churn_mensal_agg['MesAnoChurnPeriod'].dt.to_timestamp()
        df_metricas_churn = pd.merge(df_metricas_churn, churn_mensal_agg[['MesAno', 'Churn_Count']], on='MesAno', how='left')
    else:
        df_metricas_churn['Churn_Count'] = 0
    df_metricas_churn['Churn_Count'].fillna(0, inplace=True)
    df_metricas_churn['Churn_Count'] = df_metricas_churn['Churn_Count'].astype(int)
    if df_mrr_mensal_para_ativos is not None and not df_mrr_mensal_para_ativos.empty and 'MesAno' in df_mrr_mensal_para_ativos.columns:
        df_mrr_temp_ativos = df_mrr_mensal_para_ativos[['MesAno', 'Clientes_Ativos_Count']].copy()
        df_mrr_temp_ativos['MesAno'] = pd.to_datetime(df_mrr_temp_ativos['MesAno'])
        df_metricas_churn = pd.merge(df_metricas_churn, df_mrr_temp_ativos, on='MesAno', how='left')
        df_metricas_churn.rename(columns={'Clientes_Ativos_Count': 'Ativos_Inicio_Mes_Estimado'}, inplace=True)
        df_metricas_churn['Ativos_Inicio_Mes_Estimado'].fillna(0, inplace=True)
    else:
        df_metricas_churn['Ativos_Inicio_Mes_Estimado'] = 0
    df_metricas_churn['Ativos_Inicio_Mes_Estimado'] = df_metricas_churn['Ativos_Inicio_Mes_Estimado'].astype(int)
    df_metricas_churn['Taxa_Churn_Mensal'] = np.where(
        df_metricas_churn['Ativos_Inicio_Mes_Estimado'] > 0,
        (df_metricas_churn['Churn_Count'] / df_metricas_churn['Ativos_Inicio_Mes_Estimado']) * 100,
        0
    )
    df_metricas_churn['MesAnoStr'] = df_metricas_churn['MesAno'].dt.strftime('%Y-%m')
    return df_metricas_churn.sort_values(by='MesAno')

def calcular_dinamica_assinantes_mensal(df_faturas_completo, df_clientes_unicos_status_entrada_churn, data_final_base):
    if df_faturas_completo is None or df_faturas_completo.empty or \
       df_clientes_unicos_status_entrada_churn is None or df_clientes_unicos_status_entrada_churn.empty:
        return pd.DataFrame(columns=['MesAno', 'Novos', 'Cancelados', 'Ativos_Legado', 'Ativos_Totais_Mes', 'MesAnoStr'])
    df_clientes_unicos_status_entrada_churn['data_entrada_cliente'] = pd.to_datetime(df_clientes_unicos_status_entrada_churn['data_entrada_cliente'])
    df_clientes_unicos_status_entrada_churn['data_churn_cliente'] = pd.to_datetime(df_clientes_unicos_status_entrada_churn['data_churn_cliente'])
    df_faturas_completo['data_emissao'] = pd.to_datetime(df_faturas_completo['data_emissao'])
    df_faturas_completo['data_fim_estimada_fatura'] = pd.to_datetime(df_faturas_completo['data_fim_estimada_fatura'])
    min_data_ref = pd.NaT
    datas_entrada_validas = df_clientes_unicos_status_entrada_churn['data_entrada_cliente'].dropna()
    datas_emissao_validas = df_faturas_completo['data_emissao'].dropna()
    if not datas_entrada_validas.empty: min_data_ref = datas_entrada_validas.min()
    elif not datas_emissao_validas.empty: min_data_ref = datas_emissao_validas.min()
    if pd.isna(min_data_ref) or pd.isna(data_final_base):
        return pd.DataFrame(columns=['MesAno', 'Novos', 'Cancelados', 'Ativos_Legado', 'Ativos_Totais_Mes', 'MesAnoStr'])
    meses_analise = pd.date_range(start=min_data_ref.replace(day=1), end=data_final_base.replace(day=1), freq='MS')
    if meses_analise.empty and pd.notna(min_data_ref):
         meses_analise = pd.date_range(start=min_data_ref.replace(day=1), end=min_data_ref.replace(day=1), freq='MS')
    if meses_analise.empty:
        return pd.DataFrame(columns=['MesAno', 'Novos', 'Cancelados', 'Ativos_Legado', 'Ativos_Totais_Mes', 'MesAnoStr'])
    dinamica_mensal_data = []
    for mes_inicio in meses_analise:
        mes_fim = mes_inicio + pd.offsets.MonthEnd(0)
        mes_periodo = mes_inicio.to_period('M')
        novos_no_mes_df = df_clientes_unicos_status_entrada_churn[
            pd.to_datetime(df_clientes_unicos_status_entrada_churn['data_entrada_cliente']).dt.to_period('M') == mes_periodo
        ]
        count_novos = novos_no_mes_df['documento_cliente'].nunique()
        cancelados_no_mes_df = df_clientes_unicos_status_entrada_churn[
            pd.to_datetime(df_clientes_unicos_status_entrada_churn['data_churn_cliente']).dt.to_period('M') == mes_periodo
        ]
        count_cancelados = cancelados_no_mes_df['documento_cliente'].nunique()
        clientes_com_fatura_ativa_no_mes_docs = df_faturas_completo[
            (df_faturas_completo['data_emissao'] <= mes_fim) &
            (df_faturas_completo['data_fim_estimada_fatura'] >= mes_inicio)
        ]['documento_cliente'].unique()
        count_ativos_totais_mes = 0
        if len(clientes_com_fatura_ativa_no_mes_docs) > 0:
            clientes_ativos_potenciais = df_clientes_unicos_status_entrada_churn[
                df_clientes_unicos_status_entrada_churn['documento_cliente'].isin(clientes_com_fatura_ativa_no_mes_docs)
            ]
            clientes_realmente_ativos_no_mes = clientes_ativos_potenciais[
                (clientes_ativos_potenciais['data_entrada_cliente'] <= mes_fim) &
                (pd.isna(clientes_ativos_potenciais['data_churn_cliente']) | (clientes_ativos_potenciais['data_churn_cliente'] > mes_fim))
            ]
            count_ativos_totais_mes = clientes_realmente_ativos_no_mes['documento_cliente'].nunique()
        count_ativos_legado = max(0, count_ativos_totais_mes - count_novos)
        dinamica_mensal_data.append({
            'MesAno': mes_inicio, 'Novos': count_novos, 'Cancelados': count_cancelados,
            'Ativos_Legado': count_ativos_legado, 'Ativos_Totais_Mes': count_ativos_totais_mes,
            'MesAnoStr': mes_inicio.strftime('%Y-%m')
        })
    df_dinamica = pd.DataFrame(dinamica_mensal_data)
    return df_dinamica.sort_values(by='MesAno')

def calcular_receita_churn_mensal_kpi(df_faturas_completo_com_status, data_final_base):
    if df_faturas_completo_com_status is None or df_faturas_completo_com_status.empty:
        return pd.DataFrame(columns=['MesAno', 'Receita_Perdida_Churn', 'MesAnoStr'])
    df_churn_events = df_faturas_completo_com_status[
        (df_faturas_completo_com_status['status_cliente'] == 'Inativo (Churned)') &
        pd.notna(df_faturas_completo_com_status['data_churn_cliente']) &
        pd.notna(df_faturas_completo_com_status['valor_mensal_plano']) &
        (df_faturas_completo_com_status['valor_mensal_plano'] > 0)
    ].copy()
    if df_churn_events.empty:
        min_data_geral_rec_churn = pd.NaT
        if not df_faturas_completo_com_status['data_emissao'].dropna().empty:
            min_data_geral_rec_churn = df_faturas_completo_com_status['data_emissao'].dropna().min()
        if pd.notna(min_data_geral_rec_churn) and pd.notna(data_final_base):
            todos_os_meses_range_vazio_rec = pd.date_range(start=min_data_geral_rec_churn.replace(day=1),
                                                end=data_final_base.replace(day=1),
                                                freq='MS')
            if todos_os_meses_range_vazio_rec.empty and pd.notna(min_data_geral_rec_churn):
                todos_os_meses_range_vazio_rec = pd.date_range(start=min_data_geral_rec_churn.replace(day=1),
                                                               end=min_data_geral_rec_churn.replace(day=1),
                                                               freq='MS')
            df_vazio_com_meses_rec = pd.DataFrame({'MesAno': todos_os_meses_range_vazio_rec})
            if not df_vazio_com_meses_rec.empty:
                df_vazio_com_meses_rec['Receita_Perdida_Churn'] = 0
                df_vazio_com_meses_rec['MesAnoStr'] = df_vazio_com_meses_rec['MesAno'].dt.strftime('%Y-%m')
            else:
                 return pd.DataFrame(columns=['MesAno', 'Receita_Perdida_Churn', 'MesAnoStr'])
            return df_vazio_com_meses_rec
        else:
            return pd.DataFrame(columns=['MesAno', 'Receita_Perdida_Churn', 'MesAnoStr'])
    df_churn_events_unicos_por_evento = df_churn_events.sort_values(by='data_emissao', ascending=False).drop_duplicates(subset=['documento_cliente', 'data_churn_cliente'], keep='first')
    df_churn_events_unicos_por_evento['data_churn_cliente_dt'] = pd.to_datetime(df_churn_events_unicos_por_evento['data_churn_cliente'])
    df_churn_events_unicos_por_evento['MesAnoChurn'] = df_churn_events_unicos_por_evento['data_churn_cliente_dt'].dt.to_period('M')
    receita_churn_mensal_agg = df_churn_events_unicos_por_evento.groupby('MesAnoChurn')['valor_mensal_plano'].sum().reset_index(name='Receita_Perdida_Churn')
    receita_churn_mensal_agg.rename(columns={'MesAnoChurn': 'MesAnoPeriod'}, inplace=True)
    receita_churn_mensal_agg['MesAno'] = receita_churn_mensal_agg['MesAnoPeriod'].dt.to_timestamp()
    min_data_ref_rec_churn = pd.NaT
    if not df_faturas_completo_com_status['data_emissao'].dropna().empty:
        min_data_ref_rec_churn = df_faturas_completo_com_status['data_emissao'].dropna().min()
    if pd.isna(min_data_ref_rec_churn) and not receita_churn_mensal_agg.empty:
        min_data_ref_rec_churn = receita_churn_mensal_agg['MesAno'].min()
    elif pd.isna(min_data_ref_rec_churn):
        return pd.DataFrame(columns=['MesAno', 'Receita_Perdida_Churn', 'MesAnoStr'])
    todos_meses_rec_churn = pd.date_range(start=min_data_ref_rec_churn.replace(day=1),
                                          end=data_final_base.replace(day=1),
                                          freq='MS')
    if todos_meses_rec_churn.empty and pd.notna(min_data_ref_rec_churn):
         todos_meses_rec_churn = pd.date_range(start=min_data_ref_rec_churn.replace(day=1),
                                               end=min_data_ref_rec_churn.replace(day=1),
                                               freq='MS')
    if todos_meses_rec_churn.empty:
        return pd.DataFrame(columns=['MesAno', 'Receita_Perdida_Churn', 'MesAnoStr'])
    df_todos_meses_rec_churn = pd.DataFrame({'MesAno': todos_meses_rec_churn})
    df_receita_churn_final = pd.merge(df_todos_meses_rec_churn, receita_churn_mensal_agg[['MesAno', 'Receita_Perdida_Churn']], on='MesAno', how='left').fillna(0)
    df_receita_churn_final['Receita_Perdida_Churn'] = df_receita_churn_final['Receita_Perdida_Churn'].astype(float).round(2)
    df_receita_churn_final['MesAnoStr'] = df_receita_churn_final['MesAno'].dt.strftime('%Y-%m')
    return df_receita_churn_final.sort_values(by='MesAno')

def calcular_cohort_retencao_heatmap(df_faturas_pipeline, df_clientes_unicos_para_cohort, data_final_analise):
    if df_faturas_pipeline is None or df_faturas_pipeline.empty or \
       df_clientes_unicos_para_cohort is None or df_clientes_unicos_para_cohort.empty or \
       'data_entrada_cliente' not in df_clientes_unicos_para_cohort.columns:
        return pd.DataFrame(), pd.Series(dtype='int')
    for col in ['data_entrada_cliente', 'data_churn_cliente']:
        if col in df_clientes_unicos_para_cohort.columns:
            df_clientes_unicos_para_cohort[col] = pd.to_datetime(df_clientes_unicos_para_cohort[col], errors='coerce')
    for col in ['data_emissao', 'data_fim_estimada_fatura']:
        if col in df_faturas_pipeline.columns:
            df_faturas_pipeline[col] = pd.to_datetime(df_faturas_pipeline[col], errors='coerce')
        else:
            return pd.DataFrame(), pd.Series(dtype='int')
    df_clientes_unicos_para_cohort.dropna(subset=['data_entrada_cliente'], inplace=True)
    if df_clientes_unicos_para_cohort.empty: return pd.DataFrame(), pd.Series(dtype='int')
    df_clientes_unicos_para_cohort['safra_periodo'] = df_clientes_unicos_para_cohort['data_entrada_cliente'].dt.to_period('M')
    min_safra_dt = df_clientes_unicos_para_cohort['data_entrada_cliente'].min()
    max_analise_dt = pd.to_datetime(data_final_analise)
    if pd.isna(min_safra_dt): return pd.DataFrame(), pd.Series(dtype='int')
    cohort_data = []
    safra_initial_size_df = df_clientes_unicos_para_cohort.groupby('safra_periodo')['documento_cliente'].nunique().reset_index(name='tamanho_inicial_safra')
    for _, row_cliente in df_clientes_unicos_para_cohort.iterrows():
        doc = row_cliente['documento_cliente']
        safra = row_cliente['safra_periodo']
        data_entrada_cliente_dt = row_cliente['data_entrada_cliente']
        data_churn_cliente_dt = row_cliente['data_churn_cliente']
        status_cliente_final_val = row_cliente['status_cliente']
        data_fim_efetiva_servico_dt = pd.NaT
        if status_cliente_final_val == 'Ativo':
            faturas_do_cliente_ativo = df_faturas_pipeline[df_faturas_pipeline['documento_cliente'] == doc]
            if not faturas_do_cliente_ativo.empty:
                data_fim_efetiva_servico_dt = faturas_do_cliente_ativo['data_fim_estimada_fatura'].max()
        elif status_cliente_final_val == 'Inativo (Churned)' and pd.notna(data_churn_cliente_dt):
            data_fim_efetiva_servico_dt = data_churn_cliente_dt
        if pd.isna(data_fim_efetiva_servico_dt) or pd.isna(data_entrada_cliente_dt):
             continue
        mes_atual_cohort_inicio_dt = data_entrada_cliente_dt.replace(day=1)
        mes_desde_entrada_val = 0
        while mes_atual_cohort_inicio_dt <= max_analise_dt and mes_atual_cohort_inicio_dt <= data_fim_efetiva_servico_dt:
            if mes_atual_cohort_inicio_dt >= data_entrada_cliente_dt.replace(day=1) and \
               (mes_atual_cohort_inicio_dt + pd.offsets.MonthEnd(0)) <= (data_fim_efetiva_servico_dt + pd.offsets.MonthEnd(0)) :
                cohort_data.append({
                    'safra_periodo': safra,
                    'mes_desde_entrada': mes_desde_entrada_val,
                    'documento_cliente': doc
                })
            if data_fim_efetiva_servico_dt < (mes_atual_cohort_inicio_dt + pd.offsets.MonthBegin(1)):
                break
            mes_atual_cohort_inicio_dt = mes_atual_cohort_inicio_dt + pd.offsets.MonthBegin(1)
            mes_desde_entrada_val += 1
            if mes_desde_entrada_val > 24 : break
    if not cohort_data: return pd.DataFrame(), pd.Series(dtype='int')
    df_cohort_full = pd.DataFrame(cohort_data).drop_duplicates()
    cohort_counts = df_cohort_full.groupby(['safra_periodo', 'mes_desde_entrada'])['documento_cliente'].nunique().reset_index(name='clientes_ativos')
    cohort_retencao = pd.merge(cohort_counts, safra_initial_size_df, on='safra_periodo', how='left')
    cohort_retencao = cohort_retencao[cohort_retencao['tamanho_inicial_safra'] > 0]
    if cohort_retencao.empty: return pd.DataFrame(), pd.Series(dtype='int')
    cohort_retencao['taxa_retencao'] = (cohort_retencao['clientes_ativos'] / cohort_retencao['tamanho_inicial_safra']) * 100
    heatmap_data = cohort_retencao.pivot_table(index='safra_periodo', columns='mes_desde_entrada', values='taxa_retencao')
    if heatmap_data.empty: return pd.DataFrame(), pd.Series(dtype='int')
    heatmap_data.index = heatmap_data.index.strftime('%Y-%m')
    heatmap_data.columns = [f"M{col}" for col in heatmap_data.columns]
    safra_initial_size_df['safra_periodo_str'] = safra_initial_size_df['safra_periodo'].dt.strftime('%Y-%m')
    safra_initial_size_map = safra_initial_size_df.set_index('safra_periodo_str')['tamanho_inicial_safra']
    return heatmap_data.fillna(np.nan), safra_initial_size_map

def calcular_cohort_ltv_acumulado_heatmap(df_faturas_pipeline, df_clientes_unicos_para_cohort, data_final_analise):
    if df_faturas_pipeline is None or df_faturas_pipeline.empty or \
       df_clientes_unicos_para_cohort is None or df_clientes_unicos_para_cohort.empty or \
       'data_entrada_cliente' not in df_clientes_unicos_para_cohort.columns or \
       'documento_cliente' not in df_faturas_pipeline.columns or \
       'data_emissao' not in df_faturas_pipeline.columns or \
       'valor_total' not in df_faturas_pipeline.columns:
        return pd.DataFrame(), pd.Series(dtype='int')
    df_clientes_unicos_para_cohort['data_entrada_cliente'] = pd.to_datetime(df_clientes_unicos_para_cohort['data_entrada_cliente'], errors='coerce')
    df_faturas_pipeline['data_emissao'] = pd.to_datetime(df_faturas_pipeline['data_emissao'], errors='coerce')
    df_faturas_pipeline.dropna(subset=['data_emissao', 'valor_total', 'documento_cliente'], inplace=True)
    df_clientes_unicos_para_cohort.dropna(subset=['data_entrada_cliente'], inplace=True)
    if df_clientes_unicos_para_cohort.empty: return pd.DataFrame(), pd.Series(dtype='int')
    df_clientes_unicos_para_cohort['safra_periodo'] = df_clientes_unicos_para_cohort['data_entrada_cliente'].dt.to_period('M')
    min_safra_dt = df_clientes_unicos_para_cohort['data_entrada_cliente'].min()
    max_analise_dt = pd.to_datetime(data_final_analise)
    if pd.isna(min_safra_dt): return pd.DataFrame(), pd.Series(dtype='int')
    cohort_ltv_data = []
    safra_initial_size_df_ltv = df_clientes_unicos_para_cohort.groupby('safra_periodo')['documento_cliente'].nunique().reset_index(name='tamanho_inicial_safra')
    for _, row_cliente in df_clientes_unicos_para_cohort.iterrows():
        doc = row_cliente['documento_cliente']
        safra = row_cliente['safra_periodo']
        data_entrada_cliente_dt = row_cliente['data_entrada_cliente']
        faturas_do_cliente = df_faturas_pipeline[
            (df_faturas_pipeline['documento_cliente'] == doc) &
            (df_faturas_pipeline['data_emissao'] >= data_entrada_cliente_dt)
        ].sort_values(by='data_emissao')
        if faturas_do_cliente.empty: continue
        mes_atual_cohort_inicio_dt = data_entrada_cliente_dt.replace(day=1)
        mes_desde_entrada_val = 0
        while mes_atual_cohort_inicio_dt <= max_analise_dt:
            fim_do_mes_atual_cohort = mes_atual_cohort_inicio_dt + pd.offsets.MonthEnd(0)
            faturas_ate_este_mes_cohort = faturas_do_cliente[
                faturas_do_cliente['data_emissao'] <= fim_do_mes_atual_cohort
            ]
            ltv_acumulado_neste_ponto = faturas_ate_este_mes_cohort['valor_total'].sum()
            cohort_ltv_data.append({
                'safra_periodo': safra,
                'mes_desde_entrada': mes_desde_entrada_val,
                'documento_cliente': doc,
                'ltv_acumulado_cliente': ltv_acumulado_neste_ponto
            })
            data_ultima_fatura_cliente_dt = faturas_do_cliente['data_emissao'].max()
            if pd.notna(data_ultima_fatura_cliente_dt) and data_ultima_fatura_cliente_dt < (mes_atual_cohort_inicio_dt + pd.offsets.MonthBegin(1)):
                if mes_desde_entrada_val > 0 and ltv_acumulado_neste_ponto == faturas_do_cliente['valor_total'].sum():
                    break
            mes_atual_cohort_inicio_dt = mes_atual_cohort_inicio_dt + pd.offsets.MonthBegin(1)
            mes_desde_entrada_val += 1
            if mes_desde_entrada_val > 24: break
    if not cohort_ltv_data: return pd.DataFrame(), pd.Series(dtype='int')
    df_cohort_ltv_full = pd.DataFrame(cohort_ltv_data)
    cohort_ltv_agg = df_cohort_ltv_full.groupby(['safra_periodo', 'mes_desde_entrada'])['ltv_acumulado_cliente'].mean().reset_index(name='ltv_medio_acumulado')
    heatmap_ltv_data = cohort_ltv_agg.pivot_table(index='safra_periodo', columns='mes_desde_entrada', values='ltv_medio_acumulado')
    if heatmap_ltv_data.empty: return pd.DataFrame(), pd.Series(dtype='int')
    heatmap_ltv_data.index = heatmap_ltv_data.index.strftime('%Y-%m')
    heatmap_ltv_data.columns = [f"M{col}" for col in heatmap_ltv_data.columns]
    safra_initial_size_df_ltv['safra_periodo_str'] = safra_initial_size_df_ltv['safra_periodo'].dt.strftime('%Y-%m')
    safra_initial_size_map_ltv = safra_initial_size_df_ltv.set_index('safra_periodo_str')['tamanho_inicial_safra']
    return heatmap_ltv_data.fillna(np.nan), safra_initial_size_map_ltv

@st.cache_data
def processar_pipeline_completo(_df_bruto, _correcoes_map, _data_referencia_para_churn):
    df_bruto_copia = _df_bruto.copy()
    correcoes_map_copia = _correcoes_map.copy()
    df_processado_inicial = processar_dados_iniciais(df_bruto_copia)
    if df_processado_inicial is None: return None
    df_com_map_inicial = aplicar_mapeamentos_ao_df(df_processado_inicial, correcoes_map_copia)
    df_com_metricas_fatura = calcular_metricas_fatura(df_com_map_inicial)
    df_com_metricas_cliente = adicionar_metricas_cliente(df_com_metricas_fatura)
    df_final_pipeline = analisar_ciclo_de_vida_clientes(df_com_metricas_cliente, _data_referencia_para_churn)
    return df_final_pipeline

# CHAMA O FORMULÁRIO DE LOGIN E VERIFICA A AUTENTICAÇÃO
if not login_form():
    st.stop() # Impede a execução do restante do script se o login não for bem-sucedido

# SE CHEGOU AQUI, O USUÁRIO ESTÁ AUTENTICADO
st.sidebar.success(f"Autenticado como: {os.environ.get(APP_USERNAME_ENV_VAR, 'admin')}")
st.sidebar.button("Sair", on_click=logout, key="logout_button")
st.sidebar.markdown("---") # Adiciona um separador

# --- INÍCIO DA INTERFACE STREAMLIT REORGANIZADA ---

st.title("🚀 Edupulses B2B Dashboard - Análise de Clientes")
st.caption(f"Data/Hora Atual: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# --- Definição das Abas ---
tab_config, tab_financeiro, tab_clientes, tab_churn = st.tabs([
    "⚙️ Configuração e Ajustes",
    "💰 Visão Geral Financeira",
    "👥 Análise de Clientes & Retenção",
    "📉 Análise de Churn & Dinâmica"
])


# --- ABA 1: LÓGICA DE CARREGAMENTO E AJUSTES ---
with tab_config:
    st.header("1. Carregar e Processar a Base de Dados")

    # Inicialização do session_state (originalmente no topo)
    if 'df_pipeline_output' not in st.session_state: st.session_state.df_pipeline_output = None
    if 'correcoes_aplicadas_map' not in st.session_state: st.session_state.correcoes_aplicadas_map = {}
    if 'last_uploaded_filename' not in st.session_state: st.session_state.last_uploaded_filename = None
    if 'mapeamentos_carregados_do_arquivo' not in st.session_state: st.session_state.mapeamentos_carregados_do_arquivo = False
    if 'mapa_foi_alterado_na_ui' not in st.session_state: st.session_state.mapa_foi_alterado_na_ui = False

    # Carregamento inicial do arquivo de mapeamento (originalmente no topo)
    if not st.session_state.mapeamentos_carregados_do_arquivo:
        mapeamentos_salvos_dict = carregar_mapeamentos_salvos(GCP_BUCKET_NAME, GCP_FILE_PATH)
        if mapeamentos_salvos_dict:
            st.session_state.correcoes_aplicadas_map.update(mapeamentos_salvos_dict)
            st.sidebar.success(f"Mapeamentos carregados de '{NOME_ARQUIVO_MAPEAMENTO}'!")
        else:
            st.sidebar.info(f"Arquivo de mapeamento '{NOME_ARQUIVO_MAPEAMENTO}' não encontrado.")
        st.session_state.mapeamentos_carregados_do_arquivo = True

    # Upload do arquivo (originalmente na sidebar)
    arquivo_base_b2b = st.file_uploader("Carregue a base B2B (.csv ou .xlsx)", type=["csv", "xlsx", "xls"], key="uploader_base")

    # Lógica de processamento do pipeline (originalmente no topo)
    if arquivo_base_b2b is not None:
        if st.session_state.df_pipeline_output is None or \
           st.session_state.last_uploaded_filename != arquivo_base_b2b.name or \
           st.session_state.mapa_foi_alterado_na_ui:

            st.session_state.last_uploaded_filename = arquivo_base_b2b.name
            df_bruto_upload = carregar_dados(arquivo_base_b2b)

            if df_bruto_upload is not None:
                data_ref_churn_pipeline = datetime.now()
                if not df_bruto_upload.empty and 'Data de emissão' in df_bruto_upload.columns:
                     datas_emissao_validas_bruto = pd.to_datetime(df_bruto_upload['Data de emissão'], errors='coerce').dropna()
                     if not datas_emissao_validas_bruto.empty:
                         data_ref_churn_pipeline = datas_emissao_validas_bruto.max()

                mapa_correcoes_para_pipeline = st.session_state.correcoes_aplicadas_map.copy()

                with st.spinner("Processando dados... Isso pode levar um momento."):
                    st.session_state.df_pipeline_output = processar_pipeline_completo(df_bruto_upload, mapa_correcoes_para_pipeline, data_ref_churn_pipeline)

                if st.session_state.df_pipeline_output is not None:
                    st.success("Pipeline de dados completo executado.")
                else:
                    st.error("O processamento dos dados falhou.")
                st.session_state.mapa_foi_alterado_na_ui = False
            else:
                st.session_state.df_pipeline_output = None

# --- Verificação global se os dados foram carregados antes de popular as outras abas ---
if st.session_state.df_pipeline_output is None:
    with tab_config:
        st.info("Aguardando o upload de um arquivo para iniciar a análise.")
    with tab_financeiro: st.info("Faça o upload de um arquivo na aba '⚙️ Configuração e Ajustes' para começar.")
    with tab_clientes: st.info("Faça o upload de um arquivo na aba '⚙️ Configuração e Ajustes' para começar.")
    with tab_churn: st.info("Faça o upload de um arquivo na aba '⚙️ Configuração e Ajustes' para começar.")
    st.stop()


# --- LÓGICA DE FILTROS NA SIDEBAR (SEU CÓDIGO ORIGINAL, 100% PRESERVADO) ---
st.sidebar.markdown("---")
st.sidebar.header("🔎 Filtros Gerais")
data_inicio_default_filtro, data_fim_default_filtro, min_data_permitida_filtro, max_data_permitida_filtro = None, None, None, None
opcoes_tipo_cliente_filtro, opcoes_plano_filtro = ["Todos"], ["Todos"]
opcoes_periodicidade_filtro = ["Todos"] + PERIODICIDADE_VALIDAS
opcoes_status_filtro = ["Todos"] + STATUS_CLIENTE_VALIDOS
data_referencia_kpis_indicador_geral = datetime.now().replace(day=1) - timedelta(days=1)

df_completo_para_filtros = st.session_state.df_pipeline_output

if 'data_emissao' in df_completo_para_filtros.columns:
    datas_emissao_validas_base = df_completo_para_filtros['data_emissao'].dropna()
    if not datas_emissao_validas_base.empty:
        min_data_permitida_filtro = datas_emissao_validas_base.min()
        max_data_base = datas_emissao_validas_base.max()
        data_inicio_default_filtro = min_data_permitida_filtro
        data_fim_default_filtro = max_data_base
        max_data_permitida_filtro = max_data_base
        st.sidebar.caption(f"Período da base: {min_data_permitida_filtro.strftime('%d/%m/%Y')} a {max_data_base.strftime('%d/%m/%Y')}")
        primeiro_dia_mes_max_data_base = max_data_base.replace(day=1)
        data_referencia_kpis_indicador_geral = (primeiro_dia_mes_max_data_base - timedelta(days=1))

if 'tipo_cliente' in df_completo_para_filtros.columns:
    tipos_cliente_validos = [str(tc) for tc in df_completo_para_filtros['tipo_cliente'].dropna().unique() if tc]
    opcoes_tipo_cliente_filtro = ["Todos"] + sorted(list(set(tipos_cliente_validos)))
if 'nome_plano_extraido' in df_completo_para_filtros.columns:
    planos_validos_base_filtro = [str(p) for p in df_completo_para_filtros['nome_plano_extraido'].dropna().unique() if p]
    opcoes_plano_filtro = ["Todos"] + sorted(list(set(planos_validos_base_filtro)))
if 'status_cliente' in df_completo_para_filtros.columns:
    status_validos_base_filtro = [str(s) for s in df_completo_para_filtros['status_cliente'].dropna().unique() if s]
    opcoes_status_filtro = ["Todos"] + sorted(list(set(status_validos_base_filtro)))

if data_inicio_default_filtro and data_fim_default_filtro:
    data_inicio_selecionada = st.sidebar.date_input("Data Início Análise (Emissão)", value=data_inicio_default_filtro,
                                                    min_value=min_data_permitida_filtro, max_value=max_data_permitida_filtro,
                                                    key="filtro_data_inicio_val_v3")
    data_fim_selecionada = st.sidebar.date_input("Data Fim Análise (Emissão)", value=data_fim_default_filtro,
                                                 min_value=min_data_permitida_filtro, max_value=max_data_permitida_filtro,
                                                 key="filtro_data_fim_val_v3")
    if data_inicio_selecionada > data_fim_selecionada:
        st.sidebar.error("Data de início não pode ser maior que a data de fim.")
        data_inicio_selecionada = data_inicio_default_filtro
        data_fim_selecionada = data_fim_default_filtro
else:
    data_inicio_selecionada, data_fim_selecionada = None, None

tipo_cliente_selecionado_filtro_val = st.sidebar.selectbox("Tipo de Cliente", options=opcoes_tipo_cliente_filtro, key="filtro_tipo_cliente_val_sidebar_v3")
plano_selecionado_filtro_val = st.sidebar.selectbox("Tipo de Plano", options=opcoes_plano_filtro, key="filtro_plano_val_sidebar_v3")
periodicidade_selecionada_filtro_val = st.sidebar.selectbox("Periodicidade do Plano", options=opcoes_periodicidade_filtro, key="filtro_periodicidade_val_sidebar_v3")
status_selecionado_filtro_val = st.sidebar.selectbox("Status do Cliente", options=opcoes_status_filtro, key="filtro_status_val_sidebar_v3")

df_filtrado_para_kpis = st.session_state.df_pipeline_output.copy()

if data_inicio_selecionada and data_fim_selecionada:
    data_inicio_ts, data_fim_ts = pd.Timestamp(data_inicio_selecionada), pd.Timestamp(data_fim_selecionada)
    df_filtrado_para_kpis = df_filtrado_para_kpis[
        (df_filtrado_para_kpis['data_emissao'] >= data_inicio_ts) &
        (df_filtrado_para_kpis['data_emissao'] <= data_fim_ts)
    ]

if tipo_cliente_selecionado_filtro_val != "Todos": df_filtrado_para_kpis = df_filtrado_para_kpis[df_filtrado_para_kpis['tipo_cliente'] == tipo_cliente_selecionado_filtro_val]
if plano_selecionado_filtro_val != "Todos": df_filtrado_para_kpis = df_filtrado_para_kpis[df_filtrado_para_kpis['nome_plano_extraido'] == plano_selecionado_filtro_val]
if periodicidade_selecionada_filtro_val != "Todos": df_filtrado_para_kpis = df_filtrado_para_kpis[df_filtrado_para_kpis['periodicidade_extraida'] == periodicidade_selecionada_filtro_val]
if status_selecionado_filtro_val != "Todos": df_filtrado_para_kpis = df_filtrado_para_kpis[df_filtrado_para_kpis['status_cliente'] == status_selecionado_filtro_val]

if data_fim_selecionada:
    primeiro_dia_mes_data_fim_filtro = pd.Timestamp(data_fim_selecionada).replace(day=1)
    data_referencia_kpis_final_indicador = (primeiro_dia_mes_data_fim_filtro - timedelta(days=1))
else:
    data_referencia_kpis_final_indicador = data_referencia_kpis_indicador_geral


# --- POPULANDO AS ABAS COM SEU CÓDIGO ORIGINAL ---

with tab_financeiro:
    # --- KPIs de MRR ---
    st.subheader("📈 Monthly Recurring Revenue (MRR) & ARPA")
    st.caption(f"""
    **O que é:**
    - **MRR (Receita Recorrente Mensal):** Receita previsível mensal.
    - **ARPA (Receita Média Por Conta):** Receita média por cliente ativo no mês.
    **Como é calculado:**
    - MRR: Soma do `valor_mensal_plano` de faturas ativas no mês.
    - ARPA: `MRR Total Mensal` / `Número de Clientes Ativos Únicos no Mês`.
    """)
    df_mrr_calculado_kpi = pd.DataFrame()
    if not df_filtrado_para_kpis.empty:
        df_mrr_calculado_kpi = calcular_mrr_arpa_mensal(df_filtrado_para_kpis)
        if not df_mrr_calculado_kpi.empty:
            ultimo_mes_fechado_para_indicador_str = data_referencia_kpis_final_indicador.strftime('%Y-%m')
            dados_ultimo_mes_fechado_mrr = df_mrr_calculado_kpi[df_mrr_calculado_kpi['MesAnoStr'] == ultimo_mes_fechado_para_indicador_str]
            mrr_display_val, arpa_display_val = "N/A", "N/A"
            if not dados_ultimo_mes_fechado_mrr.empty:
                mrr_ultimo_mes_fechado = dados_ultimo_mes_fechado_mrr['MRR_Total'].iloc[0]
                arpa_ultimo_mes_fechado = dados_ultimo_mes_fechado_mrr['ARPA'].iloc[0]
                mrr_display_val, arpa_display_val = f"R$ {mrr_ultimo_mes_fechado:,.2f}", f"R$ {arpa_ultimo_mes_fechado:,.2f}"
            col1_mrr, col2_mrr = st.columns(2)
            col1_mrr.metric(f"MRR Total ({ultimo_mes_fechado_para_indicador_str})", mrr_display_val)
            col2_mrr.metric(f"ARPA ({ultimo_mes_fechado_para_indicador_str})", arpa_display_val)
            fig_mrr = px.line(df_mrr_calculado_kpi, x='MesAnoStr', y='MRR_Total', title="Evolução do MRR Total", labels={'MesAnoStr': 'Mês', 'MRR_Total': 'MRR (R$)'}, markers=True, text='MRR_Total')
            fig_mrr.update_traces(texttemplate='R$%{text:,.0f}', textposition='top center')
            fig_mrr.update_layout(yaxis_tickprefix='R$ ')
            st.plotly_chart(fig_mrr, use_container_width=True)
            if 'ARPA' in df_mrr_calculado_kpi.columns and not df_mrr_calculado_kpi['ARPA'].isnull().all() and (df_mrr_calculado_kpi['ARPA'].fillna(0) != 0).any():
                fig_arpa = px.line(df_mrr_calculado_kpi, x='MesAnoStr', y='ARPA', title="Evolução da ARPA", labels={'MesAnoStr': 'Mês', 'ARPA': 'ARPA (R$)'}, markers=True, text='ARPA')
                fig_arpa.update_traces(texttemplate='R$%{text:,.0f}', textposition='top center')
                fig_arpa.update_layout(yaxis_tickprefix='R$ ')
                st.plotly_chart(fig_arpa, use_container_width=True)
            else: st.info("Gráfico de ARPA não gerado: sem dados de ARPA válidos e não nulos/zero para exibir.")
            with st.expander("Dados de MRR e ARPA Mensal (Tabela)"):
                df_para_exibir_mrr_tabela = df_mrr_calculado_kpi.sort_values(by="MesAno", ascending=False)
                st.dataframe(df_para_exibir_mrr_tabela[['MesAnoStr', 'MRR_Total', 'Clientes_Ativos_Count', 'ARPA']])
        else: st.info("Não há dados suficientes para calcular o MRR com os filtros atuais (df_mrr_calculado_kpi vazio).")
    else: st.info("Nenhum dado encontrado com os filtros selecionados para calcular MRR (df_filtrado_para_kpis vazio).")

    st.markdown("---")
    # --- KPIs de LTV ---
    st.subheader("💰 Lifetime Value (LTV)")
    st.caption(f"""
    **O que é:** Receita total média gerada por cliente.
    **Como é calculado:** Média do `total_gasto_cliente` (soma de `valor_total` de todas as faturas) dos clientes únicos na seleção.
    **Cohort:** LTV médio por safra de entrada.
    """)
    if not df_filtrado_para_kpis.empty and 'documento_cliente' in df_filtrado_para_kpis.columns and 'total_gasto_cliente' in df_filtrado_para_kpis.columns and 'data_entrada_cliente' in df_filtrado_para_kpis.columns:
        df_clientes_unicos_ltv = df_filtrado_para_kpis.drop_duplicates(subset=['documento_cliente'])
        if not df_clientes_unicos_ltv.empty:
            ltv_medio_base = df_clientes_unicos_ltv['total_gasto_cliente'].mean()
            st.metric("LTV Médio da Base (Seleção Atual)", f"R$ {ltv_medio_base:,.2f}" if pd.notna(ltv_medio_base) else "N/A")
            st.markdown("---")
            st.markdown("##### LTV Médio por Safra de Entrada")
            if not df_clientes_unicos_ltv['data_entrada_cliente'].dropna().empty:
                df_ltv_cohort_calc_ltv = df_clientes_unicos_ltv.copy()
                df_ltv_cohort_calc_ltv['safra'] = pd.to_datetime(df_ltv_cohort_calc_ltv['data_entrada_cliente']).dt.to_period('M')
                cohort_data_ltv = df_ltv_cohort_calc_ltv.groupby('safra')['total_gasto_cliente'].agg(['count', 'mean']).reset_index()
                cohort_data_ltv.rename(columns={'count': 'Num_Clientes_Safra', 'mean': 'LTV_Medio_Safra'}, inplace=True)
                cohort_data_ltv['safra'] = cohort_data_ltv['safra'].astype(str)
                if not cohort_data_ltv.empty:
                    fig_cohort_ltv = px.bar(cohort_data_ltv.sort_values(by='safra'), x='safra', y='LTV_Medio_Safra', text='LTV_Medio_Safra', title='LTV Médio por Safra de Entrada (R$)')
                    fig_cohort_ltv.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
                    fig_cohort_ltv.update_layout(yaxis_tickprefix='R$ ', uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_cohort_ltv, use_container_width=True)
                    with st.expander("Dados de LTV por Safra (Tabela da Seleção)"):
                        st.dataframe(cohort_data_ltv[['safra', 'Num_Clientes_Safra', 'LTV_Medio_Safra']].sort_values(by="safra", ascending=False))
                else:
                    st.info("Não há dados de safra suficientes para o gráfico de cohort de LTV.")
            else:
                st.info("Coluna 'data_entrada_cliente' ausente ou vazia para gerar cohort de LTV.")
        else: st.info("Não há clientes únicos suficientes na seleção para calcular LTV.")
    else: st.info("Nenhum dado ou colunas necessárias encontradas com os filtros selecionados para calcular LTV.")


with tab_clientes:
    # --- KPI de Tempo Médio de Permanência ---
    st.subheader("⏳ Tempo Médio de Permanência")
    st.caption(f"""
    **O que é:** Indica, em média, por quantos meses os clientes permanecem com um contrato ativo com a empresa.
    **Como é calculado:** Para cada cliente, diferença entre `data_fim_relacionamento` (data do churn ou fim do contrato ativo) e `data_entrada_cliente`, convertido para meses. Inclui tempo futuro para ativos.
    """)
    if not df_filtrado_para_kpis.empty:
        df_permanencia_clientes_calc = calcular_metricas_permanencia(df_filtrado_para_kpis)
        if not df_permanencia_clientes_calc.empty and 'permanencia_meses' in df_permanencia_clientes_calc.columns:
            tempo_medio_geral_meses = df_permanencia_clientes_calc['permanencia_meses'].mean()
            tempo_medio_ativos_meses = df_permanencia_clientes_calc[df_permanencia_clientes_calc['status_cliente'] == 'Ativo']['permanencia_meses'].mean()
            col1_perm, col2_perm = st.columns(2)
            col1_perm.metric("Tempo Médio Permanência (Geral da Seleção, meses)", f"{tempo_medio_geral_meses:.1f}" if pd.notna(tempo_medio_geral_meses) else "N/A")
            col2_perm.metric("Tempo Médio Permanência (Ativos da Seleção, meses)", f"{tempo_medio_ativos_meses:.1f}" if pd.notna(tempo_medio_ativos_meses) else "N/A")
            with st.expander("Dados de Permanência por Cliente (Tabela da Seleção)"):
                st.dataframe(df_permanencia_clientes_calc[['documento_cliente', 'data_entrada_cliente', 'status_cliente', 'data_fim_relacionamento', 'permanencia_meses']].sort_values(by="permanencia_meses", ascending=False))
            st.markdown("---")
            st.markdown("##### Análise de Cohort de Permanência Média (por Safra da Seleção)")
            if 'data_entrada_cliente' in df_permanencia_clientes_calc.columns and not df_permanencia_clientes_calc['data_entrada_cliente'].dropna().empty:
                df_perm_cohort_calc = df_permanencia_clientes_calc.copy()
                df_perm_cohort_calc['safra'] = pd.to_datetime(df_perm_cohort_calc['data_entrada_cliente']).dt.to_period('M')
                cohort_data_permanencia = df_perm_cohort_calc.groupby('safra')['permanencia_meses'].agg(['count', 'mean']).reset_index()
                cohort_data_permanencia.rename(columns={'count': 'Num_Clientes_Safra', 'mean': 'Permanencia_Media_Meses'}, inplace=True)
                cohort_data_permanencia['safra'] = cohort_data_permanencia['safra'].astype(str)
                if not cohort_data_permanencia.empty:
                    fig_cohort_permanencia = px.bar(cohort_data_permanencia.sort_values(by='safra'), x='safra', y='Permanencia_Media_Meses', text='Permanencia_Media_Meses', title='Permanência Média por Safra de Entrada (Meses)')
                    fig_cohort_permanencia.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig_cohort_permanencia.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_cohort_permanencia, use_container_width=True)
                else: st.info("Não há dados de safra suficientes para o gráfico de cohort de permanência.")
            else: st.info("Coluna 'data_entrada_cliente' ausente ou vazia para gerar cohort de permanência.")
        else: st.info("Sem dados para métricas de permanência com filtros atuais.")
    else: st.info("Sem dados para métricas de permanência com filtros atuais.")

    # --- HEATMAP DE RETENÇÃO ---
    st.markdown("---")
    st.markdown("##### Heatmap de Retenção de Clientes por Safra (%)")
    st.caption(f"""
    **O que é:** Mostra o percentual de clientes de uma safra (mês de entrada) que permanecem ativos ao longo dos meses subsequentes.
    **Como é calculado:** Para cada safra e cada "mês desde a entrada": (Clientes da safra ainda ativos no mês X) / (Total de clientes na safra inicial).
    O tamanho inicial da safra (N) é mostrado na primeira coluna (M0).
    """)
    if not df_filtrado_para_kpis.empty:
        data_final_heatmap_calc = data_referencia_kpis_final_indicador + pd.offsets.MonthEnd(0)
        if data_fim_selecionada:
            data_final_heatmap_calc = pd.Timestamp(data_fim_selecionada)
        elif 'data_emissao' in df_filtrado_para_kpis and not df_filtrado_para_kpis['data_emissao'].dropna().empty:
             max_data_emissao_heatmap = df_filtrado_para_kpis['data_emissao'].max()
             if pd.notna(max_data_emissao_heatmap): data_final_heatmap_calc = max_data_emissao_heatmap
        df_clientes_unicos_para_heatmap = df_filtrado_para_kpis.drop_duplicates(subset=['documento_cliente'])[['documento_cliente', 'data_entrada_cliente', 'status_cliente', 'data_churn_cliente']].copy()
        heatmap_data_retencao, safra_initial_size_map_retencao = calcular_cohort_retencao_heatmap(st.session_state.df_pipeline_output, df_clientes_unicos_para_heatmap, data_final_heatmap_calc)
        if not heatmap_data_retencao.empty:
            z_text = []
            for safra_idx, safra_nome in enumerate(heatmap_data_retencao.index):
                row_text = []
                tamanho_inicial = safra_initial_size_map_retencao.get(safra_nome, 0)
                for mes_idx, mes_nome_col in enumerate(heatmap_data_retencao.columns):
                    valor_celula = heatmap_data_retencao.iloc[safra_idx, mes_idx]
                    if pd.notna(valor_celula):
                        if mes_idx == 0: row_text.append(f"{valor_celula:.0f}%<br>(N={tamanho_inicial})")
                        elif valor_celula > 0: row_text.append(f"{valor_celula:.0f}%")
                        else: row_text.append("")
                    else: row_text.append("")
                z_text.append(row_text)
            fig_heatmap_retencao = go.Figure(data=go.Heatmap(z=heatmap_data_retencao.values, x=heatmap_data_retencao.columns, y=heatmap_data_retencao.index, colorscale='Greens', text=z_text, texttemplate="%{text}", hoverongaps=False, xgap=1, ygap=1))
            fig_heatmap_retencao.update_layout(title='Taxa de Retenção de Clientes por Safra', xaxis_title='Meses Desde a Entrada', yaxis_title='Safra de Entrada (Ano-Mês)', yaxis_autorange='reversed', height=max(400, len(heatmap_data_retencao.index) * 35 + 100))
            st.plotly_chart(fig_heatmap_retencao, use_container_width=True)
        else: st.info("Não foi possível gerar o heatmap de retenção com os dados e filtros atuais.")
    else: st.info("Dados de permanência não disponíveis para o heatmap de retenção.")

    # --- HEATMAP DE LTV ACUMULADO ---
    st.markdown("---")
    st.markdown("##### Heatmap de LTV Acumulado por Safra (R$)")
    st.caption(f"""
    **O que é:** Mostra o LTV médio acumulado para os clientes de uma safra ao longo dos meses subsequentes à sua entrada.
    **Como é calculado:** Para cada safra e "mês desde a entrada": Soma da receita (`valor_total` das faturas) gerada pelos clientes da safra até aquele mês X, dividido pelo total de clientes na safra inicial.
    A célula M0 mostra o LTV médio no primeiro mês e o tamanho inicial da safra (N).
    """)
    if not df_filtrado_para_kpis.empty:
        data_final_heatmap_ltv = data_referencia_kpis_final_indicador + pd.offsets.MonthEnd(0)
        if data_fim_selecionada:
            data_final_heatmap_ltv = pd.Timestamp(data_fim_selecionada)
        elif 'data_emissao' in df_filtrado_para_kpis and not df_filtrado_para_kpis['data_emissao'].dropna().empty:
             max_data_emissao_heatmap_ltv = df_filtrado_para_kpis['data_emissao'].max()
             if pd.notna(max_data_emissao_heatmap_ltv): data_final_heatmap_ltv = max_data_emissao_heatmap_ltv
        df_clientes_unicos_ltv_heatmap = df_filtrado_para_kpis.drop_duplicates(subset=['documento_cliente'])
        heatmap_data_ltv_acum, safra_initial_size_map_ltv = calcular_cohort_ltv_acumulado_heatmap(df_filtrado_para_kpis, df_clientes_unicos_ltv_heatmap, data_final_heatmap_ltv)
        if not heatmap_data_ltv_acum.empty:
            z_text_ltv = []
            for safra_idx_ltv, safra_nome_ltv in enumerate(heatmap_data_ltv_acum.index):
                row_text_ltv = []
                tamanho_inicial_ltv = safra_initial_size_map_ltv.get(safra_nome_ltv, 0)
                for mes_idx_ltv, mes_nome_col_ltv in enumerate(heatmap_data_ltv_acum.columns):
                    valor_celula_ltv = heatmap_data_ltv_acum.iloc[safra_idx_ltv, mes_idx_ltv]
                    if pd.notna(valor_celula_ltv):
                        if mes_idx_ltv == 0: row_text_ltv.append(f"R${valor_celula_ltv:,.0f}<br>(N={tamanho_inicial_ltv})")
                        elif valor_celula_ltv > 0: row_text_ltv.append(f"R${valor_celula_ltv:,.0f}")
                        else: row_text_ltv.append("")
                    else: row_text_ltv.append("")
                z_text_ltv.append(row_text_ltv)
            fig_heatmap_ltv = go.Figure(data=go.Heatmap(z=heatmap_data_ltv_acum.values, x=heatmap_data_ltv_acum.columns, y=heatmap_data_ltv_acum.index, colorscale='Blues', text=z_text_ltv, texttemplate="%{text}", hoverongaps=False, xgap=1, ygap=1))
            fig_heatmap_ltv.update_layout(title='LTV Médio Acumulado por Safra', xaxis_title='Meses Desde a Entrada', yaxis_title='Safra de Entrada (Ano-Mês)', yaxis_autorange='reversed', height=max(400, len(heatmap_data_ltv_acum.index) * 35 + 100))
            st.plotly_chart(fig_heatmap_ltv, use_container_width=True)
        else: st.info("Não foi possível gerar o heatmap de LTV acumulado com os dados e filtros atuais.")
    else: st.info("Dados de LTV não disponíveis para o heatmap.")


with tab_churn:
    # --- KPIs de Churn ---
    st.subheader("📉 Churn de Clientes")
    st.caption(f"""
    **O que é:** Perda de clientes.
    - **Quantidade:** Nº de clientes que se tornaram 'Inativo (Churned)' no mês. O churn ocorre se não há renovação em 90 dias após fim da fatura.
    - **Taxa Mensal:** (Qtd Churn / Ativos no Início do Mês Estimado) * 100.
    **Cálculo:** `data_churn_cliente` é a data de fim da fatura que não foi renovada. `Ativos no Início do Mês Estimado` vem do cálculo do MRR.
    """)
    df_mrr_para_taxa_churn = calcular_mrr_arpa_mensal(df_filtrado_para_kpis)
    if not df_filtrado_para_kpis.empty:
        df_clientes_unicos_para_churn_kpi = df_filtrado_para_kpis.drop_duplicates(subset=['documento_cliente'])
        data_final_base_para_churn_recente = data_referencia_kpis_final_indicador + pd.offsets.MonthEnd(0)
        if data_fim_selecionada:
            data_final_base_para_churn_recente = pd.Timestamp(data_fim_selecionada)
        elif 'data_emissao' in df_filtrado_para_kpis and not df_filtrado_para_kpis['data_emissao'].dropna().empty:
            max_data_emissao_filtrada = df_filtrado_para_kpis['data_emissao'].max()
            if pd.notna(max_data_emissao_filtrada): data_final_base_para_churn_recente = max_data_emissao_filtrada
        df_churn_mensal_calculado = calcular_metricas_churn_mensal(df_clientes_unicos_para_churn_kpi, data_final_base_para_churn_recente, df_mrr_para_taxa_churn)
        if not df_churn_mensal_calculado.empty:
            ultimo_mes_fechado_para_indicador_str_churn = data_referencia_kpis_final_indicador.strftime('%Y-%m')
            churn_ultimo_mes_df = df_churn_mensal_calculado[df_churn_mensal_calculado['MesAnoStr'] == ultimo_mes_fechado_para_indicador_str_churn]
            churn_count_ultimo_mes_val, churn_rate_ultimo_mes_val = 0, 0.0
            if not churn_ultimo_mes_df.empty:
                churn_count_ultimo_mes_val = churn_ultimo_mes_df['Churn_Count'].iloc[0]
                churn_rate_ultimo_mes_val = churn_ultimo_mes_df['Taxa_Churn_Mensal'].iloc[0]
            col1_churn, col2_churn = st.columns(2)
            col1_churn.metric(f"Quantidade de Churn ({ultimo_mes_fechado_para_indicador_str_churn})", churn_count_ultimo_mes_val)
            col2_churn.metric(f"Taxa de Churn ({ultimo_mes_fechado_para_indicador_str_churn})", f"{churn_rate_ultimo_mes_val:.2f}%" if pd.notna(churn_rate_ultimo_mes_val) else "N/A")
            fig_churn_qtd = px.bar(df_churn_mensal_calculado, x='MesAnoStr', y='Churn_Count', title="Evolução da Quantidade de Churn Mensal", labels={'MesAnoStr': 'Mês', 'Churn_Count': 'Nº de Clientes Churned'}, text='Churn_Count')
            fig_churn_qtd.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_churn_qtd, use_container_width=True)
            fig_churn_taxa = px.line(df_churn_mensal_calculado, x='MesAnoStr', y='Taxa_Churn_Mensal', title="Evolução da Taxa de Churn Mensal (%)", labels={'MesAnoStr': 'Mês', 'Taxa_Churn_Mensal': 'Taxa de Churn (%)'}, markers=True, text='Taxa_Churn_Mensal')
            fig_churn_taxa.update_traces(texttemplate='%{text:.1f}%', textposition='top center')
            fig_churn_taxa.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_churn_taxa, use_container_width=True)
            with st.expander("Dados de Churn Mensal (Tabela da Seleção)"):
                df_para_exibir_churn = df_churn_mensal_calculado.sort_values(by="MesAno", ascending=False)
                st.dataframe(df_para_exibir_churn[['MesAnoStr', 'Churn_Count', 'Ativos_Inicio_Mes_Estimado', 'Taxa_Churn_Mensal']])
        else: st.info("Não há dados de churn para exibir com os filtros atuais.")

        st.markdown("---")
        st.markdown("##### Clientes Churned nos Últimos 90 Dias")
        data_inicio_churn_recente_calc = data_final_base_para_churn_recente - timedelta(days=90)
        churns_recentes_df_calc = df_clientes_unicos_para_churn_kpi[(df_clientes_unicos_para_churn_kpi['status_cliente'] == 'Inativo (Churned)') & (pd.to_datetime(df_clientes_unicos_para_churn_kpi['data_churn_cliente']) >= data_inicio_churn_recente_calc) & (pd.to_datetime(df_clientes_unicos_para_churn_kpi['data_churn_cliente']) <= data_final_base_para_churn_recente)].copy()
        if not churns_recentes_df_calc.empty:
            if 'nome_cliente' not in churns_recentes_df_calc.columns and 'nome_cliente' in st.session_state.df_pipeline_output.columns:
                 nomes_clientes_map = st.session_state.df_pipeline_output.drop_duplicates(subset=['documento_cliente'])[['documento_cliente', 'nome_cliente']].set_index('documento_cliente')
                 churns_recentes_df_calc['nome_cliente'] = churns_recentes_df_calc['documento_cliente'].map(nomes_clientes_map['nome_cliente'])
            st.dataframe(churns_recentes_df_calc[['nome_cliente', 'documento_cliente', 'data_churn_cliente']].sort_values(by='data_churn_cliente', ascending=False).reset_index(drop=True))
        else: st.info(f"Nenhum cliente churnou entre {data_inicio_churn_recente_calc.strftime('%d/%m/%Y')} e {data_final_base_para_churn_recente.strftime('%d/%m/%Y')}.")
    else: st.info("Nenhum dado encontrado com os filtros selecionados para calcular Churn.")

    st.markdown("---")
    # --- Gráfico de Dinâmica de Assinantes ---
    st.subheader("📊 Dinâmica de Assinantes")
    st.caption(f"""
    **O que é:** Movimentação da base de assinantes.
    **Como é calculado (mês a mês):**
    - `Novos`: Clientes com `data_entrada_cliente` no mês.
    - `Cancelados`: Clientes com `data_churn_cliente` no mês (barras negativas).
    - `Ativos Totais`: Clientes com fatura ativa no mês, que entraram antes/durante e não churnaram antes do fim do mês.
    - `Ativos_Legado`: `Ativos Totais` - `Novos`.
    """)
    if not df_filtrado_para_kpis.empty:
        df_clientes_unicos_para_dinamica = df_filtrado_para_kpis.drop_duplicates(subset=['documento_cliente'])[['documento_cliente', 'data_entrada_cliente', 'data_churn_cliente', 'status_cliente']]
        data_final_base_dinamica = data_referencia_kpis_final_indicador + pd.offsets.MonthEnd(0)
        if data_fim_selecionada:
            data_final_base_dinamica = pd.Timestamp(data_fim_selecionada)
        elif 'data_emissao' in df_filtrado_para_kpis and not df_filtrado_para_kpis['data_emissao'].dropna().empty:
            max_data_emissao_filtrada_dinamica = df_filtrado_para_kpis['data_emissao'].max()
            if pd.notna(max_data_emissao_filtrada_dinamica): data_final_base_dinamica = max_data_emissao_filtrada_dinamica
        df_dinamica_assinantes_calc = calcular_dinamica_assinantes_mensal(df_filtrado_para_kpis, df_clientes_unicos_para_dinamica, data_final_base_dinamica)
        if not df_dinamica_assinantes_calc.empty:
            df_dinamica_plot_prep = df_dinamica_assinantes_calc.copy()
            df_dinamica_plot_prep['Cancelados_Plot_Val'] = df_dinamica_plot_prep['Cancelados'] * -1
            df_dinamica_plot = df_dinamica_plot_prep.melt(id_vars=['MesAnoStr'], value_vars=['Novos', 'Ativos_Legado', 'Cancelados_Plot_Val'], var_name='Tipo_Movimentacao_Melt', value_name='Numero_Clientes')
            def definir_legenda_e_rotulo(row):
                if row['Tipo_Movimentacao_Melt'] == 'Cancelados_Plot_Val': return 'Cancelados', abs(row['Numero_Clientes'])
                return row['Tipo_Movimentacao_Melt'], row['Numero_Clientes']
            res = df_dinamica_plot.apply(definir_legenda_e_rotulo, axis=1, result_type='expand')
            df_dinamica_plot['Tipo_Movimentacao_Legenda'], df_dinamica_plot['Texto_Rotulo'] = res[0], res[1].astype(int)
            ordem_legenda = {'Novos': 1, 'Ativos_Legado': 2, 'Cancelados': 3}
            df_dinamica_plot['Ordem'] = df_dinamica_plot['Tipo_Movimentacao_Legenda'].map(ordem_legenda)
            df_dinamica_plot.sort_values(by=['MesAnoStr', 'Ordem'], inplace=True)
            fig_dinamica = px.bar(df_dinamica_plot, x='MesAnoStr', y='Numero_Clientes', color='Tipo_Movimentacao_Legenda', title="Dinâmica Mensal de Assinantes", labels={'MesAnoStr': 'Mês', 'Numero_Clientes': 'Número de Clientes', 'Tipo_Movimentacao_Legenda': 'Status'}, color_discrete_map={'Novos': 'mediumseagreen', 'Ativos_Legado': 'royalblue', 'Cancelados': 'indianred'}, text='Texto_Rotulo')
            fig_dinamica.update_traces(texttemplate='%{text}', textposition='auto')
            fig_dinamica.update_layout(barmode='relative', uniformtext_minsize=8)
            st.plotly_chart(fig_dinamica, use_container_width=True)
            with st.expander("Dados da Dinâmica de Assinantes (Tabela da Seleção)"):
                st.dataframe(df_dinamica_assinantes_calc[['MesAnoStr', 'Novos', 'Ativos_Legado', 'Cancelados', 'Ativos_Totais_Mes']].sort_values(by="MesAnoStr", ascending=False))
        else: st.info("Não há dados suficientes para gerar o gráfico de dinâmica de assinantes.")
    else: st.info("Nenhum dado encontrado com os filtros selecionados para a dinâmica de assinantes.")

    st.markdown("---")
    # --- Receita Churn por Mês ---
    st.subheader("💸 Receita Churn por Mês")
    st.caption(f"""
    **O que é:** Soma do `valor_mensal_plano` de clientes que churnaram no mês.
    **Como é calculado:** Para clientes 'Inativo (Churned)', o `valor_mensal_plano` de sua última fatura ativa é somado no mês da `data_churn_cliente`.
    """)
    if not df_filtrado_para_kpis.empty:
        data_final_base_rec_churn = data_referencia_kpis_final_indicador + pd.offsets.MonthEnd(0)
        if data_fim_selecionada: data_final_base_rec_churn = pd.Timestamp(data_fim_selecionada)
        elif 'data_emissao' in df_filtrado_para_kpis and not df_filtrado_para_kpis['data_emissao'].dropna().empty:
            max_data_emissao_filtrada_rec_churn = df_filtrado_para_kpis['data_emissao'].max()
            if pd.notna(max_data_emissao_filtrada_rec_churn): data_final_base_rec_churn = max_data_emissao_filtrada_rec_churn
        df_receita_churn_calc = calcular_receita_churn_mensal_kpi(df_filtrado_para_kpis, data_final_base_rec_churn)
        if not df_receita_churn_calc.empty:
            fig_receita_churn = px.bar(df_receita_churn_calc, x='MesAnoStr', y='Receita_Perdida_Churn', title="Evolução da Receita Perdida por Churn (Mensal)", labels={'MesAnoStr': 'Mês do Churn', 'Receita_Perdida_Churn': 'Receita Perdida (R$)'}, text='Receita_Perdida_Churn')
            fig_receita_churn.update_traces(texttemplate='R$%{text:,.0f}', textposition='outside')
            fig_receita_churn.update_layout(yaxis_tickprefix='R$ ')
            st.plotly_chart(fig_receita_churn, use_container_width=True)
            with st.expander("Dados de Receita Churn Mensal (Tabela da Seleção)"):
                df_para_exibir_receita_churn = df_receita_churn_calc.sort_values(by="MesAno", ascending=False)
                st.dataframe(df_para_exibir_receita_churn[['MesAnoStr', 'Receita_Perdida_Churn']])
        else: st.info("Não há dados de receita churn para exibir com os filtros atuais.")
    else: st.info("Nenhum dado encontrado com os filtros selecionados para calcular Receita Churn.")


# --- Conteúdo da Aba de Configuração (continuação) ---
with tab_config:
    if st.session_state.df_pipeline_output is not None:
#        ...
#        with st.expander("🔧 Ajustar Classificações de Serviço"):
            df_base_para_ajuste = st.session_state.df_pipeline_output.copy()

            # Esta é a sua lógica original para df_para_ajuste_ui, que inclui itens para revisão
            df_para_ajuste_ui_parte1 = df_base_para_ajuste[~df_base_para_ajuste['periodicidade_extraida'].isin(PERIODICIDADE_VALIDAS)]
            df_para_ajuste_ui_parte2 = df_base_para_ajuste[df_base_para_ajuste['plano_identificado'] == False]
            df_para_ajuste_ui = pd.concat([df_para_ajuste_ui_parte1, df_para_ajuste_ui_parte2]).drop_duplicates(subset=['descricao_servico']).reset_index(drop=True)

            if st.session_state.correcoes_aplicadas_map:
                desc_no_mapa = list(st.session_state.correcoes_aplicadas_map.keys())
                df_no_mapa = df_base_para_ajuste[df_base_para_ajuste['descricao_servico'].isin(desc_no_mapa)]
                df_para_ajuste_ui = pd.concat([df_para_ajuste_ui, df_no_mapa]).drop_duplicates(subset=['descricao_servico'], keep='first').reset_index(drop=True)

            if not df_para_ajuste_ui.empty:
                descricoes_unicas_para_ajuste = df_para_ajuste_ui['descricao_servico'].unique()

                # Mensagem informativa sobre o total de itens na lista de ajuste/revisão
                st.info(f"Total de descrições na lista de ajuste/revisão: {len(descricoes_unicas_para_ajuste)}")

                # Modificação no st.slider
                num_descricoes_a_mostrar = st.slider(
                    "Número de descrições para exibir por vez:",
                    min_value=1,
                    # max_value agora permite selecionar até o total de itens únicos
                    max_value=max(1, len(descricoes_unicas_para_ajuste)),
                    # value inicial pode ser um número fixo ou o total, se menor
                    value=min(10, max(1, len(descricoes_unicas_para_ajuste))),
                    key="slider_num_desc_v8_full_list" # Nova chave para evitar conflitos
                )
                st.write(f"Exibindo {num_descricoes_a_mostrar} de {len(descricoes_unicas_para_ajuste)} descrições da lista de ajuste/revisão:")

                correcoes_atuais_formulario = {}
                form_correcao = st.form(key="form_correcao_servicos_v8_full_list") # Nova chave
                with form_correcao:
                    for i, desc_servico in enumerate(descricoes_unicas_para_ajuste[:num_descricoes_a_mostrar]):
                        # Pega o status atual do item diretamente do df_base_para_ajuste
                        # Este DataFrame reflete o resultado do pipeline com o mapa de correções mais recente.
                        item_df_original_pipeline = df_base_para_ajuste[df_base_para_ajuste['descricao_servico'] == desc_servico]

                        precisa_de_correcao_urgente = False
                        if not item_df_original_pipeline.empty:
                            status_item = item_df_original_pipeline.iloc[0]
                            precisa_de_correcao_urgente = (status_item['plano_identificado'] == False) or \
                                                       (status_item['periodicidade_extraida'] not in PERIODICIDADE_VALIDAS)

                        # Monta a string de descrição com ou sem destaque
                        descricao_display_md = f"**Descrição:** `{desc_servico}`"
                        if precisa_de_correcao_urgente:
                            descricao_display_md += " ⚠️ **<span style='color:orange; font-weight:bold;'>REVISÃO URGENTE</span>**"
                        else:
                            descricao_display_md += " (✔️ Mapeado)"

                        st.markdown(descricao_display_md, unsafe_allow_html=True)

                        # O restante da sua lógica de exibição de exemplos e campos do formulário continua aqui...
                        # Exemplo: (usando item_df_original_pipeline que já pegamos)
                        exemplos_info = []
                        for _, row in item_df_original_pipeline.head(3).iterrows(): # Usar item_df_original_pipeline aqui
                            nome = row.get('nome_cliente', 'N/A')
                            valor = row.get('valor_total')
                            valor_str = f"R${valor:.2f}" if pd.notna(valor) else "N/A"
                            exemplos_info.append(f"{nome} (Valor: {valor_str})")
                        if exemplos_info: st.caption(f"Exemplos: {'; '.join(exemplos_info)}")
                        else: st.caption("Nenhum exemplo.")

                        mapa_sessao_para_desc = st.session_state.correcoes_aplicadas_map.get(desc_servico, {})

                        # Sugestões baseadas no estado ATUAL do item no pipeline (df_base_para_ajuste)
                        plano_sugerido_no_df = item_df_original_pipeline['nome_plano_extraido'].iloc[0] if not item_df_original_pipeline.empty else "Sem plano definido"
                        periodicidade_sugerida_no_df = item_df_original_pipeline['periodicidade_extraida'].iloc[0] if not item_df_original_pipeline.empty else "Mensal"

                        plano_sugerido = mapa_sessao_para_desc.get('plano', plano_sugerido_no_df)
                        periodicidade_sugerida = mapa_sessao_para_desc.get('periodicidade', periodicidade_sugerida_no_df)

                        if periodicidade_sugerida not in PERIODICIDADE_VALIDAS: periodicidade_sugerida = "Mensal"

                        cols = st.columns(2)
                        chave_base_widget = re.sub(r'[^A-Za-z0-9_]', '', str(desc_servico))[:20]
                        # Atualize as chaves dos widgets para garantir que sejam únicas se a descrição for muito longa ou similar
                        key_plano = f"plano_{i}_{chave_base_widget}_v8_full_list"
                        key_period = f"period_{i}_{chave_base_widget}_v8_full_list"

                        try: plano_idx = PLANOS_DROPDOWN_OPCOES.index(plano_sugerido)
                        except ValueError: plano_idx = 0
                        novo_plano = cols[0].selectbox(f"Plano '{desc_servico[:20]}...'", options=PLANOS_DROPDOWN_OPCOES, index=plano_idx, key=key_plano)

                        try: period_idx = PERIODICIDADE_DROPDOWN_OPCOES.index(periodicidade_sugerida)
                        except ValueError: period_idx = PERIODICIDADE_DROPDOWN_OPCOES.index("Mensal")
                        nova_periodicidade = cols[1].selectbox(f"Periodicidade '{desc_servico[:20]}...'", options=PERIODICIDADE_DROPDOWN_OPCOES, index=period_idx, key=key_period)

                        correcoes_atuais_formulario[desc_servico] = {'plano': novo_plano, 'periodicidade': nova_periodicidade}
                        st.markdown("---")

                    submitted = st.form_submit_button("Aplicar Correções da Página e Atualizar Mapa da Sessão")
                if submitted:
                    mudanca_real_no_mapa = False
                    for desc, corr_nova in correcoes_atuais_formulario.items():
                        if corr_nova['periodicidade'] not in PERIODICIDADE_VALIDAS: corr_nova['periodicidade'] = "Mensal"
                        corr_antiga = st.session_state.correcoes_aplicadas_map.get(desc)
                        if corr_antiga != corr_nova :
                            st.session_state.correcoes_aplicadas_map[desc] = corr_nova
                            mudanca_real_no_mapa = True
                    if mudanca_real_no_mapa:
                            st.session_state.mapa_foi_alterado_na_ui = True
                            st.success("Mapa de correções da sessão atualizado. Recarregando para aplicar...")
                            st.rerun()
                    else: st.info("Nenhuma alteração detectada nas correções desta página.")
            else:
                st.success("🎉 Todas as descrições de serviço parecem ter uma periodicidade válida e/ou um plano identificado!")

        # --- Resumo da Extração ---
            st.markdown("---")
            st.header("3. Resumo da Extração (Filtrado)")
            df_resumo = df_filtrado_para_kpis
            if 'nome_plano_extraido' in df_resumo.columns and 'periodicidade_extraida' in df_resumo.columns:
                col1_res, col2_res = st.columns(2)
            with col1_res: st.write("Contagem por Plano Extraído:"); st.dataframe(df_resumo['nome_plano_extraido'].value_counts())
            with col2_res: st.write("Contagem por Periodicidade Extraída:"); st.dataframe(df_resumo['periodicidade_extraida'].value_counts())
            st.write("Colunas Finais (após filtros):", df_resumo.columns.tolist())


# --- Download do Mapeamento na Sidebar (SEU CÓDIGO ORIGINAL) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Salvar Correções no Cloud")

if st.sidebar.button("Salvar Mapeamentos no Cloud Storage", key="save_to_gcs"):
    if st.session_state.correcoes_aplicadas_map:
        map_df_list = []
        for k, v_dict in st.session_state.correcoes_aplicadas_map.items():
            plano_val = v_dict.get('plano')
            period_val = v_dict.get('periodicidade')
            # Apenas adiciona ao CSV se houver uma correção real
            if (plano_val and plano_val != "Sem plano definido") or \
               (period_val and period_val in PERIODICIDADE_VALIDAS and period_val != "Mensal"): # Exemplo: não salvar se for apenas "Mensal" default sem plano
                map_df_list.append({
                    'descricao_original': k,
                    'plano_corrigido': plano_val,
                    'periodicidade_corrigida': period_val
                })

        if map_df_list:
            with st.spinner("Salvando mapeamentos no Cloud Storage..."):
                try:
                    map_df = pd.DataFrame(map_df_list)
                    gcs_uri = f"gs://{GCP_BUCKET_NAME}/{GCP_FILE_PATH}"

                    # Graças ao gcsfs, o pandas pode escrever diretamente no GCS
                    map_df.to_csv(gcs_uri, index=False, encoding='utf-8-sig')

                    st.sidebar.success("Mapeamento salvo com sucesso no Cloud Storage!")
                    st.toast("Mapeamento salvo na nuvem! 🎉")
                    # Pode ser útil recarregar os mapeamentos da sessão para refletir o estado salvo,
                    # ou simplesmente confiar que a próxima execução do app carregará o novo arquivo.
                    # Para forçar recarregamento na próxima execução da UI de ajuste (se necessário):
                    # st.session_state.mapeamentos_carregados_do_arquivo = False
                    # st.experimental_rerun() # Cuidado com reruns automáticos
                except Exception as e:
                    st.sidebar.error(f"Falha ao salvar no Cloud Storage: {e}")
                    st.toast(f"Erro ao salvar: {e}", icon="❌")
        else:
            st.sidebar.warning("Nenhuma correção substancial no mapa da sessão para salvar.")
    else:
        st.sidebar.info("Nenhuma correção no mapa da sessão para salvar.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido para Edupulses")