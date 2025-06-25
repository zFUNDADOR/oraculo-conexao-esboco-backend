# main.py - Servidor Backend Python com Flask

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io # Para lidar com BLOBs (embeddings)

app = Flask(__name__)
# Habilita CORS para permitir que seu frontend React (que está em uma porta/domínio diferente)
# possa fazer requisições para este backend. ESSENCIAL para desenvolvimento no Replit.
CORS(app)

# ===========================
# CONEXÃO COM BANCO DE DADOS
# ===========================
def conectar_banco(nome_banco="banco_local.db"):
    """
    Conecta-se ao banco de dados SQLite e cria a tabela 'documentos' se ela não existir.
    """
    conn = sqlite3.connect(nome_banco)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

# ===========================
# FUNÇÃO PARA SIMULAR EMBEDDING
# ===========================
def gerar_embedding_simples(texto):
    """
    Gera um embedding vetorial aleatório para um texto.
    Em um cenário real, isso seria substituído por um modelo LLM local (ex: sentence-transformers).
    O tamanho do vetor (512) é um placeholder.
    """
    np.random.seed(hash(texto) % (2**32 - 1)) # Garante que o mesmo texto gere o mesmo "embedding" simulado
    vetor = np.random.rand(512).astype(np.float32) # Usar float32 para economizar espaço e ser mais compatível
    return vetor

# ===========================
# INSERIR TEXTO + EMBEDDING
# ===========================
def inserir_documento(conn, texto):
    """
    Insere um documento no banco de dados, gerando seu embedding.
    """
    embedding = gerar_embedding_simples(texto)
    cursor = conn.cursor()
    # Converte o array numpy para bytes para armazenar como BLOB no SQLite
    cursor.execute("INSERT INTO documentos (texto, embedding) VALUES (?, ?)",
                   (texto, embedding.tobytes()))
    conn.commit()

# ===========================
# BUSCAR SIMILARES
# ===========================
def buscar_similares(conn, texto_consulta, top_k=3):
    """
    Busca documentos similares no banco de dados usando similaridade de cosseno.
    Retorna uma lista dos (texto, score) dos documentos mais similares.
    """
    embedding_consulta = gerar_embedding_simples(texto_consulta)
    cursor = conn.cursor()
    cursor.execute("SELECT id, texto, embedding FROM documentos")
    resultados = cursor.fetchall()

    textos_com_id = []
    similaridades = []

    for id_, texto, emb_bytes in resultados:
        # Converte os bytes de volta para um array numpy
        emb = np.frombuffer(emb_bytes, dtype=np.float32) # Use float32 aqui também
        # Calcula a similaridade de cosseno
        sim = cosine_similarity([embedding_consulta], [emb])[0][0]
        textos_com_id.append((id_, texto))
        similaridades.append(sim)

    # Combina textos e similaridades e ordena do mais similar para o menos
    combinados = sorted(zip(textos_com_id, similaridades), key=lambda x: x[1], reverse=True)

    # Retorna apenas os top_k resultados
    return combinados[:top_k]

# ===========================
# Rotas da API
# ===========================

@app.route('/api/analisar-seo', methods=['POST'])
def analisar_seo_data():
    """
    Endpoint da API para receber texto, processá-lo (simuladamente),
    e retornar dados para infográficos.
    """
    data = request.get_json()
    content = data.get('content', '')

    if not content:
        return jsonify({"error": "Conteúdo de texto não fornecido."}), 400

    # --- Lógica de Análise (usando as funções do seu esqueleto Python) ---
    # Conecta ao banco de dados para cada requisição (pode ser otimizado para produção)
    conn = conectar_banco()

    # Exemplo: Inserir o conteúdo para que ele possa ser "buscado"
    # Você pode querer apenas inserir textos de "treinamento" e não cada upload do usuário
    # Por enquanto, vamos inserir para simular dados no DB
    inserir_documento(conn, content)

    # Simula a busca por termos similares ou análise do conteúdo
    # Aqui você poderia usar o LLM local para extrair informações mais complexas
    # Por exemplo, vamos buscar termos "similares" ao próprio conteúdo para gerar os gráficos
    similares = buscar_similares(conn, content, top_k=5)

    # Transforma os resultados para o formato que seu frontend espera para os gráficos
    # Para o gráfico de pizza, vamos "simular" uma distribuição de palavras-chave
    # baseada na similaridade dos documentos encontrados.
    # Isso é um exemplo, a lógica real dependeria do que o LLM extrairia.
    labels_pie = [f"Doc ID {t[0]}" for (t, s) in similares]
    data_pie = [round(s * 100, 1) for (t, s) in similares] # Score como percentual

    # Se não houver documentos similares, use dados padrão ou ajuste
    if not labels_pie:
        labels_pie = ['Nenhuma Palavra-Chave', 'Dados Genéricos']
        data_pie = [50, 50]

    # Garante que os dados somam 100% ou os distribui para o gráfico de pizza
    # Se os scores não somarem 100, Chart.js ainda os plota proporcionalmente.
    # Aqui apenas um exemplo para o label e data
    keyword_distribution = {
        "labels": labels_pie,
        "datasets": [
            {
                "data": data_pie,
                "backgroundColor": [
                    'rgba(136, 192, 208, 0.8)',
                    'rgba(163, 190, 140, 0.8)',
                    'rgba(180, 142, 173, 0.8)',
                    'rgba(235, 203, 139, 0.8)',
                    'rgba(191, 97, 106, 0.8)',
                ],
                "borderColor": [
                    'rgba(136, 192, 208, 1)',
                    'rgba(163, 190, 140, 1)',
                    'rgba(180, 142, 173, 1)',
                    'rgba(235, 203, 139, 1)',
                    'rgba(191, 97, 106, 1)',
                ],
                "borderWidth": 1,
            },
        ],
    }

    # Métricas Chave (poderiam ser extraídas pelo LLM real)
    key_metrics = [
        {"label": "Total de Palavras", "value": str(len(content.split()))},
        {"label": "Documentos Similares Encontrados", "value": str(len(similares))},
        {"label": "Primeiro Doc. Similar", "value": f"ID {similares[0][0][0]} ({similares[0][1]:.2f})" if similares else "N/A"},
        {"label": "Qualidade do Texto", "value": "Excelente (Simulado)"}, # Simulado por enquanto
    ]

    conn.close() # Fecha a conexão com o banco

    # Retorna os dados no formato JSON
    return jsonify({
        "keywordDistribution": keyword_distribution,
        "keyMetrics": key_metrics
    })

# ===========================
# PONTO DE ENTRADA DO SERVIDOR FLASK
# ===========================
if __name__ == '__main__':
    # Cria o banco de dados e insere alguns documentos de teste na inicialização
    # Estes são documentos "fixos" que o chatbot poderia "conhecer"
    conn_init = conectar_banco()
    # Somente insere se o banco estiver vazio
    cursor_init = conn_init.cursor()
    cursor_init.execute("SELECT COUNT(*) FROM documentos")
    if cursor_init.fetchone()[0] == 0:
        print("Inserindo documentos de teste no banco de dados...")
        inserir_documento(conn_init, "O marketing digital é essencial para empresas hoje em dia.")
        inserir_documento(conn_init, "SEO on-page otimiza o conteúdo de uma página para motores de busca.")
        inserir_documento(conn_init, "Mapas mentais são ferramentas visuais para organizar ideias.")
        inserir_documento(conn_init, "Gerenciamento de vídeos e sua otimização para plataformas.")
        inserir_documento(conn_init, "Inteligência artificial e aprendizado de máquina estão revolucionando a análise de dados.")
        print("Documentos de teste inseridos.")
    else:
        print("Banco de dados já contém documentos. Não inserindo novos testes.")
    conn_init.close()

    # O Replit usa a variável de ambiente PORT, mas para testar localmente pode ser 5000
    # No Replit, ele detecta automaticamente o Flask e expõe a porta
    app.run(host='0.0.0.0', port=5000, debug=True)