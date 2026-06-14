import os
import random
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine
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

fake = Faker('pt_BR')


def generate_invoice_data(num_records=5000):
    """
    Gera um DataFrame com dados sintéticos de NFs, incluindo padrões de fraude.
    """
    data = []
    
    print(f"🔄 Gerando {num_records} registros sintéticos...")
    
    for _ in range(num_records):
        # Dados básicos
        razao = fake.company()
        dt_emissao = fake.date_between(start_date='-1y', end_date='today')
        categoria = random.choice(['Serviço', 'Comércio', 'Indústria'])
        
        # --- LÓGICA DE CRIAÇÃO DO PADRÃO DE FRAUDE ---
        # Definimos aleatoriamente se este registro SERÁ uma fraude (baseado nas suas regras)
        # Vamos forçar que ~10% da base seja fraude para o modelo aprender bem
        is_fraud_case = random.random() < 0.10 
        
        if is_fraud_case:
            # Padrão de Fraude Específico:
            # CNPJ começa com 99, Valor > 40k, Descrição contém "Consultoria"
            cnpj_base = str(fake.random_number(digits=12, fix_len=True))
            cnpj = f"99{cnpj_base}" # Força o início 99
            
            valor = round(random.uniform(40001, 150000), 2) # Valor alto
            descricao = "Consultoria Especializada em Otimização Fiscal e Processos"
            risco_fraude = 1 # ALVO
        
        else:
            # Padrão Normal (Aleatório)
            cnpj = fake.cnpj().replace('.', '').replace('/', '').replace('-', '')
            valor = round(random.uniform(100, 39000), 2)
            
            # Descrições variadas para não confundir o modelo
            descricoes_normais = [
                "Venda de Material de Escritório", 
                "Manutenção de Ar Condicionado", 
                "Serviço de Limpeza Predial", 
                "Aquisição de Insumos Industriais",
                "Licença de Software"
            ]
            descricao = random.choice(descricoes_normais)
            risco_fraude = 0 # ALVO

        data.append({
            'cnpj_emitente': cnpj,
            'razao_social': razao,
            'data_emissao': dt_emissao,
            'descricao_servico': descricao,
            'valor_total': valor,
            'categoria_imposto': categoria,
            'risco_fraude': risco_fraude
        })

    return pd.DataFrame(data)


def save_to_postgres(df):
    """
    Salva o DataFrame no PostgreSQL.
    """
    try:
        print("🚀 Iniciando upload para o PostgreSQL na Hostinger...")

        # if_exists='replace' recria a tabela. Em produção real, usariamos 'append'.
        # chunksize=1000 envia blocos de 1000 linhas (Bulk Insert)
        df.to_sql(
            name='notas_fiscais_treinamento',  # Apenas o nome da tabela
            schema='treinamento',              # O nome do schema separado
            con=engine, 
            if_exists='replace', 
            index=False, 
            chunksize=1000
            )
        print("✅ Dados inseridos com sucesso!")

    except Exception as e:
        print(f"❌ Erro ao conectar ou salvar no banco: {e}")


if __name__ == "__main__":
    df_notas = generate_invoice_data(5000)
    save_to_postgres(df_notas)