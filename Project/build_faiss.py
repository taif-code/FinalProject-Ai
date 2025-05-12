import os
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# تأكد من وجود الملف
json_path = "/Users/taifaladwani/Downloads/SmartTourGuide/landmarks_list2.json"
if not os.path.exists(json_path):
    raise FileNotFoundError("❌ الملف landmarks_list2.json غير موجود في مجلد data/")

# تحميل البيانات
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

# بناء المستندات
documents = [Document(page_content=item["description"]) for item in data]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# إنشاء وحفظ الفهرس
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_landmarks")

print("✅ تم إنشاء مجلد faiss_landmarks بنجاح!")
