import os
import speech_recognition as sr
from PIL import Image
import torch
import json
import faiss
from langdetect import detect

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import CLIPModel, CLIPProcessor

# ✅ إعداد المفاتيح
os.environ["OPENAI_API_KEY"] = "your_openai_key_here"
os.environ["SERPAPI_API_KEY"] = "your_serpapi_key_here"


# ✅ تحميل النماذج
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
vectordb = FAISS.load_local("faiss_landmarks", embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ✅ تحميل نموذج CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.read_index("/Users/taifaladwani/Downloads/SmartTourGuide/data/data/embeddings/index.faiss")
with open("/Users/taifaladwani/Downloads/SmartTourGuide/landmarks_list2.json") as f:
    labels = json.load(f)

# ✅ توليد البرومبت حسب اللغة
def get_prompt_template_by_lang(lang="ar"):
    if lang == "ar":
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""أنت مرشد سياحي ذكي. 
استخدم المعلومات التالية للإجابة على السؤال بدقة وبأسلوب مختصر:

📚 السياق:  
{context}

❓ السؤال:
{question}

🧠 الجواب:""")
    else:
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a smart tour guide. Use the following context to answer the user's question accurately:

📚 Context:  
{context}

❓ Question:
{question}

🧠 Answer:""")

# ✅ تحليل نص عبر RAG
def rag_tool_func(query: str):
    lang = detect(query)
    prompt = get_prompt_template_by_lang(lang)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    rag_result = qa_chain({"query": query})
    answer = rag_result.get("result", "")

    if not answer or len(answer) < 30:
        return "❌ لم يتم العثور على معلومات كافية.", lang, "Wikipedia / Google"

    return answer, lang, "Knowledge Base"

# ✅ تحويل صوت إلى نص → ثم تحليل
def audio_tool_func(audio_path: str, lang="ar"):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio, language=lang)
        print("🎤 تم تحويل الصوت إلى نص:", text)
        return rag_tool_func(text)
    except sr.UnknownValueError:
        return "❌ لم يتم فهم الصوت بشكل صحيح.", lang, "SpeechRecognition"
    except Exception as e:
        return f"❌ خطأ: {e}", lang, "Error"

# ✅ التعرف على معلم من صورة → ثم تحليل
def recognize_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs).cpu().numpy().astype('float32')
    _, indices = index.search(img_emb, k=1)
    idx = indices[0][0]
    return labels[idx] if idx < len(labels) else "Unknown Landmark"

def image_tool_func(image_path: str):
    landmark = recognize_image(image_path)
    print("🖼️ تم التعرف على المعلم:", landmark)

    if landmark == "Unknown Landmark":
        return "❌ لم يتم التعرف على المعلم.", "ar", "CLIP"
    
    return rag_tool_func(landmark)
