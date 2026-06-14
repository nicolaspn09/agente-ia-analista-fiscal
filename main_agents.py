import os
import pandas as pd
import joblib
import warnings
from textwrap import dedent
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Bibliotecas do CrewAI e LangChain
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# 1. Carregar variáveis de ambiente
# Obtém o caminho do diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
# Procura o .env a partir do diretório do script
dotenv_path = find_dotenv(os.path.join(script_dir, '.env'))
# Carrega o .env
load_dotenv(dotenv_path)

# =============================================================================
# CONFIGURAÇÃO DO LLM
# =============================================================================
# Usamos a classe nativa LLM do CrewAI.
# O segredo é o prefixo "groq/" no nome do modelo.
llm_groq = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# =============================================================================
# FERRAMENTA CUSTOMIZADA (A Ponte entre Texto e ML)
# =============================================================================
# Esta classe é como uma "Action" ou "Component" customizado no RPA.
# O Agente vai "chamar" essa classe quando precisar calcular o risco.
class FiscalRiskAnalysisTool(BaseTool):
    name: str = "Calculadora de Risco Fiscal"
    description: str = (
        "Útil para calcular a probabilidade de fraude de uma nota fiscal. "
        "Requer como entrada: valor_total (float), descricao (texto) e cnpj (texto)."
    )

    def _run(self, valor_total: float, descricao_servico: str, cnpj_emitente: str) -> str:
        """
        Método que é executado quando o Agente decide usar a ferramenta.
        """
        try:
            # 1. Carregar o modelo treinado na Fase 1
            # (Num projeto real, carregariamos isso fora da função para cache, mas aqui fica didático)
            model = joblib.load(r'C:\Users\nicol\OneDrive\Cursos online\Treinamento Python - Hashtag\Códigos\Fiscal Agent\model_risk.pkl')

            # 2. Feature Engineering (Repetir a lógica do treinamento!)
            # O modelo não entende texto, ele precisa das flags 0 ou 1 que criamos antes.
            
            # Lógica: CNPJ começa com 99?
            cnpj_suspeito = 1 if str(cnpj_emitente).startswith("99") else 0
            
            # Lógica: Descrição tem 'Consultoria'?
            is_consultoria = 1 if "consultoria" in descricao_servico.lower() else 0

            # 3. Preparar o DataFrame para o modelo (Mesma estrutura do treino)
            features = pd.DataFrame([{
                'valor_total': float(valor_total),
                'cnpj_suspeito': cnpj_suspeito,
                'is_consultoria': is_consultoria
            }])

            # 4. Previsão
            prediction = model.predict(features)[0] # 0 = Normal, 1 = Fraude

            # 5. Retorno para o Agente (em texto, pois o Agente lê texto)
            if prediction == 1:
                return "ALERTA: O modelo matemático identificou ALTO RISCO de fraude (Target 1)."
            else:
                return "OK: O modelo matemático identificou BAIXO RISCO (Target 0)."

        except Exception as e:
            return f"Erro ao calcular risco: {str(e)}"

# Instanciamos a ferramenta para dar aos agentes
risk_tool = FiscalRiskAnalysisTool()

# =============================================================================
# DEFINIÇÃO DOS AGENTES
# =============================================================================

# Agente 1: O Extrator
# Analogia RPA: O robô que faz OCR ou lê o PDF.
extractor_agent = Agent(
    role='Analista de Extração de Dados',
    goal='Extrair e estruturar dados chave da nota fiscal para análise.',
    backstory=dedent("""
        Você é um especialista em ler documentos fiscais. 
        Sua atenção aos detalhes é impecável. Você não julga, apenas relata os fatos:
        Valores, CNPJs e Descrições.
    """),
    verbose=True,      # Mostra o "pensamento" do agente no console
    allow_delegation=False,
    llm=llm_groq
)

# Agente 2: O Auditor (Este usa a Ferramenta!)
# Analogia RPA: O robô decisor que aplica as regras de negócio complexas.
auditor_agent = Agent(
    role='Auditor Fiscal Sênior',
    goal='Analisar tecnicamente se a nota é fraudulenta ou não.',
    backstory=dedent("""
        Você é um auditor implacável com 20 anos de experiência.
        Você NÃO confia apenas na sua intuição; você SEMPRE usa sua ferramenta
        "Calculadora de Risco Fiscal" para validar a nota.
        Se a ferramenta disser que é fraude, você reprova a nota imediatamente.
    """),
    tools=[risk_tool], # <--- AQUI DAMOS O "SUPER PODER" (MODELO ML) PARA ELE
    verbose=True,
    allow_delegation=False,
    llm=llm_groq
)

# Agente 3: O Gerente
# Analogia RPA: O robô que manda o e-mail final ou gera o relatório PDF.
manager_agent = Agent(
    role='Gerente de Compliance',
    goal='Gerar o relatório final executivo para a diretoria.',
    backstory=dedent("""
        Você transforma as análises técnicas do Auditor em linguagem de negócio.
        Seu relatório deve ser claro, direto e recomendar a Ação Final (Pagar ou Bloquear).
    """),
    verbose=True,
    allow_delegation=False,
    llm=llm_groq
)

# =============================================================================
# TASKS (As Tarefas)
# =============================================================================
# Aqui definimos o que cada agente vai fazer com o input recebido.

# Dados simulados de entrada (como se viesse de uma API ou Email)
# Vamos testar um caso de FRAUDE (Consultoria + Valor Alto + CNPJ 99)
nota_input = """
Nota Fiscal Eletrônica Nº 4502
Emitente: Tech Solutions Ltda (CNPJ: 99.123.456/0001-88)
Data: 20/11/2024
Descrição: Consultoria Especializada em Otimização Fiscal e Processos
Valor Total: R$ 55.000,00
"""

task1_extract = Task(
    description=f"Analise o seguinte texto da nota fiscal e extraia: CNPJ, Valor Total e Descrição. Texto: {nota_input}",
    expected_output="Um resumo estruturado com os 3 campos extraídos.",
    agent=extractor_agent
)

task2_audit = Task(
    description="""
    Utilizando os dados extraídos, use OBRIGATORIAMENTE a ferramenta 'Calculadora de Risco Fiscal'.
    Passe os parâmetros corretos para a ferramenta.
    Com base na resposta da ferramenta, dê seu veredito técnico.
    """,
    expected_output="Parecer técnico informando se é Alto Risco ou Baixo Risco e a justificativa do modelo.",
    agent=auditor_agent
)

task3_report = Task(
    description="Com base no parecer do Auditor, escreva um relatório final em formato Markdown recomendando Aprovação ou Reprovação.",
    expected_output="Relatório Executivo em Markdown.",
    agent=manager_agent
)

# =============================================================================
# CREW (A Orquestração)
# =============================================================================
# Aqui liCOMPANY_NAME tudo. É o botão "Play" do seu processo RPA.

fiscal_crew = Crew(
    agents=[extractor_agent, auditor_agent, manager_agent],
    tasks=[task1_extract, task2_audit, task3_report],
    process=Process.sequential, # Um agente espera o outro terminar
    verbose=True
)

print("🤖 Iniciando Auditoria Autônoma...")
result = fiscal_crew.kickoff()

print("\n\n########################")
print("## RELATÓRIO FINAL ##")
print("########################\n")
print(result)