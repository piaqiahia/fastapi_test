from fastapi import FastAPI,status,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ValidationError
from typing import Optional,List,Union
import logging
from dotenv import load_dotenv
#导入本地embedding服务
from embedding_service import LocalEmbeddingService

app = FastAPI(title = "AI Application API with Local Embedding", version = "1.0")

#配置日志
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

try:
    embedding_service = LocalEmbeddingService()
except Exception as e:
    logging.error(f"Failed to load embedding model:{e}")
    embedding_service = None

#模拟企业知识库文档
DOCUMENTS = [
    "人工智能是计算机科学的一个分支，它试图创建能够执行人类智能任务的机器。",
    "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
    "深度学习是机器学习的一个分支，使用神经网络进行学习。",
    "自然语言处理是人工智能的重要应用，用于处理人类语言。",
    "大模型（LLM）是当前人工智能的热点，如 GPT、通义千问等。",
]

document_vectors = []
for doc in DOCUMENTS:
    vector = embedding_service.embed_text(doc)
    document_vectors.append(vector)

def cosine_similarity(vec1,vec2): #计算两个向量的余弦相似度
    dot_product = sum(a * b for a, b in zip(vec1, vec2)) #zip将两个向量对应位置的元素配对然后计算乘积最后累加成点积
    norm1 = sum(a * a for a in vec1) ** 0.5 #计算模长，**0.5开平方
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0 #计算余弦相似度if语句边界处理，防止分母为0

def search_similar(query_vector, top_k = 1): #在文档库中搜索最相似的文档
    similarities = [
        (i, cosine_similarity(query_vector, doc_vec))#计算查询向量与每个文档向量的余弦相似度并生成i为索引的元组
        for i, doc_vec in enumerate(document_vectors)
    ]
    similarities.sort(key = lambda x : x[1], reverse = True)#lambda匿名函数 x:列表的每个元素 x[1]元组(x)中的第二个元素（相似度）
    return [DOCUMENTS[i] for i, _ in similarities[:top_k]] #for i, _  取i忽略后面的相似度 [:top_k]取排序后前 top_k 个元素

#请求/响应模型
class EmbedRequest(BaseModel):
    text : str

class EmbedBatchRequest(BaseModel):
    texts : List[str]

class EmbedResponse(BaseModel):
    vector : List[float]

class EmbedBatchResponse(BaseModel):
    vectors : List[List[float]]

class AskRequest(BaseModel):
    question : str

class AskResponse(BaseModel):
    question : str
    answer : str
    context : str

class ItemCreate(BaseModel):
    name : str
    description : Optional[str] = None #str可选，默认None
    price : float

class ItemResponse(BaseModel):
    id : int
    name : str
    description : Optional[str]
    price : float

#===== 问答接口 =====

@app.post("/ask", response_model = AskResponse)
def ask_question(request : AskRequest): #接收用户问题，返回基于文档的模拟答案
    if embedding_service is None:
        raise HTTPException(status_code = 500, detail = "Embedding model not available")

    try:
        query_vector = embedding_service.embed_text(request.question) #生成问题向量
        top_context = search_similar(query_vector, top_k = 1)[0] #搜索最相关的文档
        answer = f"根据文档内容：{top_context}" #模拟生成答案

        return AskResponse(
            question = request.question,
            answer = answer,
            context = top_context
        )
    except Exception as e:
        logging.error(f"Ask error:{str(e)}")
        raise HTTPException(status_code = 500, detail = "Answer generation failed")

#===== 新增路由：单文本嵌入 =====
@app.post("/embed", response_model = EmbedResponse)
def embed_text(request : EmbedRequest):
    if embedding_service is None:
        raise HTTPException(status_code = 500, detail = "Embedding model not available")

    try:
        vector = embedding_service.embed_text(request.text)
        return EmbedResponse(vector = vector) #EmbedResponse有个vector字段后面的vector传入前面这个
    except Exception as e:
        logging.error(f"Embedding error:{e}")
        raise HTTPException(status_code = 500, detail = "Embedding failed")

#===== 新增路由：批量嵌入 =====
@app.post("/embed/batch", response_model = EmbedBatchResponse)
def embed_batch(request : EmbedBatchRequest):
    if embedding_service is None:
        raise HTTPException(status_code = 500, detail = "Embedding model not available")

    try:
        vectors = embedding_service.embed_documents(request.texts)
        return EmbedBatchResponse(vectors = vectors)
    except Exception as e:
        logging.error(f"Batch embedding error:{e}")
        raise HTTPException(status_code = 500, detail = "Batch embedding failed")

#===== 异常处理器 =====
#1. 处理自定义 HTTPException（如 404）
@app.exception_handler(HTTPException)
async def http_exception_handler(request : Request, exc : HTTPException):
    logger.warning(f"Http error {exc.status_code} : {exc.detail}")
    return JSONResponse(
        status_code = exc.status_code,
        content = {"error" : "Request failed", "detail" : exc.detail}
    )

#2. 处理 Pydantic 验证错误（如字段缺失、类型错误）
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request : Request, exc : RequestValidationError):
    logger.error(f"validation error ： {exc.errors()}")
    return JSONResponse(
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
        content = {"error" : "Invalid input", "detail" : exc.errors()}
    )

#3. 兜底：处理未预期的服务器错误
@app.exception_handler(Exception)
async def global_exception_handler(request : Request, exc : Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content = {"error" : "Internal Server Error", "detail" : "Please contact support"}
    )
@app.get("/") #当用户用 GET 方法访问 根路径 / 时，执行下面的函数
def read_root():
    return {"message": "Welcome to FastAPI", "status": "OK"}

@app.get("/items/{item_id}") #访问 /items/42，则 item_id = 42
def read_item(item_id : int): #None，表示可选  /items/5?q=test → item_id=5, q="test" q是查询参数
    if item_id <= 0:
        raise HTTPException(status_code = 400, detail = "Item ID must be positive")
    if item_id == 999:
        raise HTTPException(status_code = 404, detail = "Item not found")
    return {"item_id" : item_id, "name" : f"Item{item_id}"}

@app.post("/items/", response_model = ItemResponse, status_code = status.HTTP_201_CREATED)
def create_item(item : ItemCreate):
    fake_id = 1001
    return ItemResponse(
        id = fake_id,
        name = item.name,
        description = item.description,
        price = item.price
    )

@app.get("/search/")
def search_items(query : str = "default", limit : int = 10):
    results = [{"id" : i, "title" : f"Result{i}"} for i in range(limit)]
    return {"query" : query, "limit" : limit, "results" : [f"Result{i}" for i in range(limit)]}

#===== 简单中间件：记录请求 =====
@app.middleware("http")
async def log_request(request : Request, call_next):
    logger.info(f"Request :{request.method}{request.url}")
    response = await call_next(request)
    logger.info(f"Request status:{response.status_code}")
    return response