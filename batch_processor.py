import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus

# 1. Configuração de Ambiente
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = find_dotenv(os.path.join(script_dir, '.env'))
load_dotenv(dotenv_path)


def get_db_connection():
    """Cria a conexão segura com o Banco (Igual ao gerador)"""
    db_user = quote_plus(os.getenv('DB_USER'))
    db_pass = quote_plus(os.getenv('DB_PASS'))
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    
    DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(DB_URL)

def generate_verdict(row):
    """
    Função auxiliar para criar o texto do parecer baseado nos dados.
    Isso simula o que o Auditor escreveria, mas de forma performática.
    """
    if row['previsao_fraude'] == 1:
        motivos = []
        if row['cnpj_suspeito'] == 1:
            motivos.append("CNPJ com prefixo de risco (99)")
        if row['is_consultoria'] == 1:
            motivos.append("Serviço sensível (Consultoria)")
        if row['valor_total'] > 40000:
            motivos.append(f"Valor atípico (R$ {row['valor_total']:.2f})")
            
        return f"REPROVADO: Alto risco detectado. Fatores: {', '.join(motivos)}."
    else:
        return "APROVADO: Transação dentro dos padrões de normalidade."
    

def run_batch_audit():
    print("🚀 Iniciando Auditoria em Lote (Batch Processing)...")
    
    # 1. Carregar Modelo
    model_path = os.path.join(script_dir, 'model_risk.pkl')
    if not os.path.exists(model_path):
        print("❌ Erro: Modelo 'model_risk.pkl' não encontrado.")
        return

    print("🧠 Carregando cérebro digital (Modelo ML)...")
    model = joblib.load(model_path)
    
    # 2. Ler dados do Banco
    print("📡 Baixando notas fiscais do PostgreSQL...")
    engine = get_db_connection()
    query = "SELECT * FROM treinamento.notas_fiscais_treinamento"
    df = pd.read_sql(query, engine)
    
    print(f"📊 Processando {len(df)} registros...")

    # 3. Feature Engineering (Preparar dados para o modelo)
    # Precisamos criar as mesmas colunas que usamos no treino!
    df_features = pd.DataFrame()
    df_features['valor_total'] = df['valor_total'].astype(float)
    df_features['cnpj_suspeito'] = df['cnpj_emitente'].astype(str).str.startswith('99').astype(int)
    df_features['is_consultoria'] = df['descricao_servico'].astype(str).str.contains('Consultoria', case=False).astype(int)
    
    # Selecionar apenas as colunas que o modelo espera
    X = df_features[['valor_total', 'cnpj_suspeito', 'is_consultoria']]

    # 4. Predição em Massa (O momento mágica)
    # O modelo analisa as 5000 linhas em fração de segundos aqui
    df['previsao_fraude'] = model.predict(X)
    
    # Adicionamos as colunas auxiliares no DF final para facilitar a leitura da sua amiga
    df['cnpj_suspeito'] = df_features['cnpj_suspeito']
    df['is_consultoria'] = df_features['is_consultoria']

    # 5. Gerar Parecer (Texto)
    print("✍️ Escrevendo pareceres técnicos...")
    df['parecer_auditoria'] = df.apply(generate_verdict, axis=1)

    # 6. Exportar para Excel
    output_file = os.path.join(script_dir, 'Relatorio_Auditoria_Final.xlsx')
    print(f"💾 Salvando arquivo Excel em: {output_file}")
    
    # Vamos formatar um pouco melhor removendo colunas técnicas desnecessárias para o negócio
    colunas_finais = [
        'data_emissao', 'razao_social', 'cnpj_emitente', 
        'descricao_servico', 'valor_total', 'parecer_auditoria', 'previsao_fraude'
    ]
    
    df[colunas_finais].to_excel(output_file, index=False)
    print("✅ Sucesso! Pode enviar para validação.")


if __name__ == "__main__":
    run_batch_audit()