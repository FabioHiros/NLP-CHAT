import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
db_uri = "mysql+mysqlconnector://root:147258369@localhost:3306/produtos"
db = SQLDatabase.from_uri(db_uri)

api_key= os.getenv('GROQ_API_KEY')


client = Groq(api_key=api_key)

chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")

template = '''
Você é um assistente de compras de celulares para a loja virtual mercado livre (https://www.mercadolivre.com.br/). Deverá receber as perguntas do usuário e trasformá-las
em uma query para um banco MySQL {schema}.
As informações relevantes podem estar tanto no título quanto na coluna do atributo, por exemplo 8gb de ram pode estar descrito na coluna
titulo (exemplos: 'huawei honor magic 2 256gb rom 8gb ram + brindes', 'celular quantum l sim 16 gb preto 8 gb ram') ou na coluna ram (exemplo:'8 gb'),
diferente da coluna titulo que não segue um padrão definido as colunas ram, memoria_interna, camera_frontal e camera_traseira estará formatada da mesma maneira em todos os casos (com um espaço entre o valor e a unidade de memória).
Além disso há as colunas preco, cor, condicao, modelo, modelo processador e marca. Segue descrições do que condiz cada coluna:
A coluna preco armazena o valor do produto em reais(BRL) como um dado DECIMAL(10,2), sendo a única informação que não pode ser encontrada no titulo.
A coluna cor pode armazenar a cor do produto, porém como no caso dos outros dados armazenados uma pesquisa na coluna titulo deve ser feita para caso esta esteja vazia.
A coluna condicao nunca estará vazia e armazena somente dois valores possíveis, 'novo' e 'usado', sendo eles autodescritíveis.
A coluna modelo contém o modelo do celular, porém também não segue um padrão, podendo estar vazio ou com nomes incompletos, é recomendado buscar na coluna titulo também caso esta coluna esteja vazia, além de buscas usando 'LIKE', por exemplo 'Samsung Galaxy s23' você buscaria somente por 's23' nesta coluna, pareando buscas por 'samsung' e 'galaxy' nas outras colunas para garantir precisão.
A coluna modelo_processador contém o modelo do processador e é improvável que seja encontrado na coluna titulo, portando uma busca simplificada usando 'LIKE' é a mais provável de encontrar resultados.
A coluna marca contém o fabricante do celular, samsung, apple, xiaomi etc. Esta coluna nunca estará vazia e possui dados confiáveis não requerendo o uso de pesquisas por 'LIKE'.
Baseado nessas informações responda APENAS com uma query MySQL e selecione os 5 primeiros aparelhos que condizem com os requisitos, confira para não errar a sintaxe, se atente nos espaçamentos das palavras fornecidas pelo usuário, por exemplo: 8gb ram = 8 gb ram = 8gbram = 8 ram.
Segue exemplos de queries corretas:
usuário: "Celular 8gb de ram"
query: SELECT * FROM celulares WHERE (titulo LIKE '%16gb ram%' OR titulo LIKE '%16 gb ram%') OR ram = '16 gb' LIMIT 5;
usuário: "Iphone barato"
query: SELECT * FROM celulares WHERE preco < 700 AND (titulo LIKE '%iphone%' OR marca = 'apple') LIMIT 5;
usuário: "Celular preto"
query: SELECT * FROM celulares WHERE cor = 'preto' OR titulo LIKE '%preto%' LIMIT 5;
usuário: "Celular usado"
query: SELECT * FROM celulares WHERE condicao = 'usado' LIMIT 5;
usuário: "Samsung Galaxy S23"
query: SELECT * FROM celulares WHERE (titulo LIKE '%galaxy%' OR marca = 'samsung') AND (modelo LIKE '%s23%' OR titulo LIKE '%s23%') LIMIT 5;
Quando a requisição for, por exemplo, quero os celulares com as melhores cameras frontais, gere uma query que procure os maiores valores na coluna camera_frontal.
Não retorne querys entre aspas ou com qualquer tipo de pontuação.
É muito importante que você lembre que as informações podem estar contidas no titulo ou na coluna especifica
Question:{question}
SQL Query:
'''

prompt = ChatPromptTemplate.from_template(template)

def get_schema(_):
    schema = db.get_table_info()
    # print(f"Schema: {schema}")
    return schema

llm = chat
out = StrOutputParser()

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop='\nSQL Query:')
    | out
)

def run_query(query):
    # print(f"Running query: {query}")
    result = db.run(query)
    # print(f"Query result: {result}")
    return result

def clean_query(query):
    cleaned_query = query.replace("\\*", "*").replace("\n", " ").strip()
    print(f"Cleaned SQL Query: {cleaned_query}")
    return cleaned_query


template2 = """Baseado na pergunta, sql query e sql response abaixo, faça uma resposta com linguagem natural falando dos produtos retornados e forneça os links, ao apresentar os resultados use esse modelo anuncio: nome do produto (baseado na coluna titulo), uma descrição curta e por fim o link: (que se encontra na coluna permalink). Responda sempre em pt-br.
Formate o texto para cada celular, de alguns espaços entre cada celular e utilize html para isso:

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

prompt_response = ChatPromptTemplate.from_template(template2)


full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        # schema=get_schema,
        response=lambda vars: run_query(clean_query(vars["query"])),
    )
    | prompt_response
    | llm
)

def get_response(user_question):
    data = str(full_chain.invoke({"question": user_question}))
    content_start_index = data.find("content='") + len("content='")
    content_end_index = data.find("response_metadata")
    content = data[content_start_index:content_end_index]
    content = content.replace("\\n", " ").replace("\n", " ").strip()
    # print(f"Response: {content}")

    return content

# print(template2)
