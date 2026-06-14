import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv, find_dotenv
from urllib.parse import quote_plus

# 1. Carregar variáveis de ambiente
# Obtém o caminho do diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
# Procura o .env a partir do diretório do script
dotenv_path = find_dotenv(os.path.join(script_dir, '.env'))
# Carrega o .env
load_dotenv(dotenv_path)

# --- CORREÇÃO DE SEGURANÇA E FORMATAÇÃO ---
# Tratamos caracteres especiais (como @, #, /) na senha e usuário
db_user = quote_plus(os.getenv('DB_USER'))
db_pass = quote_plus(os.getenv('DB_PASS'))
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# Montagem da URL com as variáveis tratadas
DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(DB_URL)

def load_data():
    """Lê os dados diretamente do PostgreSQL"""
    print("📡 Lendo dados do banco...")
    query = "SELECT * FROM treinamento.notas_fiscais_treinamento"
    
    return pd.read_sql(query, engine)


def train():
    df = load_data()

    # --- FEATURE ENGINEERING (Pré-processamento) ---
    # O modelo precisa de números. Vamos converter regras de negócio em colunas numéricas.
    
    # 1. Flag para CNPJ suspeito (começa com 99)
    #astype(int) converte True para 1 e False para 0
    df['cnpj_suspeito'] = df['cnpj_emitente'].str.startswith('99').astype(int) 
    
    # 2. Flag para palavra-chave na descrição
    df['is_consultoria'] = df['descricao_servico'].str.contains('Consultoria', case=False).astype(int)
    
    # Seleção de Features (X) e Alvo (y)
    # Não vamos passar a data ou o texto cru, apenas as características numéricas que criamos
    features = ['valor_total', 'cnpj_suspeito', 'is_consultoria']
    X = df[features]
    y = df['risco_fraude']

    # Divisão Treino e Teste (80% para aprender, 20% para prova)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- TREINAMENTO ---
    print("🧠 Treinando Random Forest...")
    # n_estimators=100 é como ter 100 "mini robôs" votando se é fraude ou não
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- AVALIAÇÃO ---
    print("\n📊 Avaliação do Modelo:")
    y_pred = model.predict(X_test)
    
    # O Recall da classe 1 é o mais importante: De todas as fraudes reais, quantas peCOMPANY_NAME?
    print(classification_report(y_test, y_pred))
    
    # --- SALVAR O MODELO ---
    joblib.dump(model, r'C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Fiscal Agent\model_risk.pkl')
    print("💾 Modelo salvo como 'model_risk.pkl'")


if __name__ == "__main__":
    train()