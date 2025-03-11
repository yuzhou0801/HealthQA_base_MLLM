import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
import whisper
from gtts import gTTS
import tempfile

# 全局变量（避免通过 gr.State 传递大型对象）
model = None
tokenizer = None
index = None
texts = None
whisper_model = None

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载文本嵌入模型
text_embedding_model = SentenceTransformer(r"D:\all-MiniLM-L6-v2")


# 加载文本检索系统
def load_text_retrieval_system(index_file, texts_file):
    """
    加载文本检索系统（FAISS 索引和文本数据）。
    """
    index = faiss.read_index(index_file)
    with open(texts_file, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return index, texts


# 检索文本
def retrieve_texts(query, index, texts, k=3, threshold=0.6):
    """ 检索与查询相关的文本，并设定相似度阈值 """
    query_embedding = text_embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding.astype("float32"), k)
    retrieved_texts = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:  # 设定相似度阈值，避免低相关性文本
            retrieved_texts.append(texts[i])
    return retrieved_texts


# 加载微调好的模型
def load_model(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    return model, tokenizer


# 加载 Whisper 语音识别模型
def load_whisper_model():
    global whisper_model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")


# 语音转文字函数
def transcribe_audio(audio_file):
    if audio_file is None:
        return ""
    # 使用 Whisper 对录音文件进行转录
    result = whisper_model.transcribe(audio_file)
    text = result["text"].strip()
    return text


# 文本转语音函数，生成临时 mp3 文件并返回文件路径
def text_to_speech(text):
    if not text:
        return None
    tts = gTTS(text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.write_to_fp(fp)
        temp_filename = fp.name
    return temp_filename


# 提取最后一条助手回复并转换为语音
def play_answer(history):
    # history 为 openai-style 格式的对话记录列表
    answer = None
    for msg in reversed(history):
        if msg.get("role") == "assistant" and msg.get("content"):
            answer = msg["content"]
            break
    if not answer:
        return None
    audio_file = text_to_speech(answer)
    return audio_file


# 切换语音播报：点击后自动生成并播放，重复点击则停止播放
def toggle_tts(chat_history, is_playing):
    if not is_playing:
        audio_file = play_answer(chat_history)
        if not audio_file:
            # 没有生成语音则保持未播放状态
            return gr.update(value=None), "Voice broadcast", False
        # 返回更新后的隐藏音频组件（设置 autoplay=True）、按钮文本更新为“停止播放”及播放状态 True
        return gr.update(value=audio_file, autoplay=True), "Stop Playing", True
    else:
        # 停止播放：清空音频组件，并将按钮文本恢复为“语音播报”
        return gr.update(value=None), "Voice broadcast", False


# 格式化用户输入
def format_query(user_input, retrieved_texts):
    """ 格式化用户输入，并确保检索文本不会干扰模型 """
    if retrieved_texts:
        context = "\n\n".join(f"Retrieved Document {i + 1}: {text}" for i, text in enumerate(retrieved_texts))
        formatted_query = (
            f"### Retrieved Context ###\n"
            f"{context}\n\n"
            f"### User Question ###\n"
            f"{user_input}\n\n"
            f"### Assistant Answer ###\n"
        )
    else:
        formatted_query = (
            f"### User Question ###\n"
            f"{user_input}\n\n"
            f"### Assistant Answer ###\n"
        )
    return formatted_query


# 生成模型回复，新增 use_rag 参数控制是否调用检索模块
def generate_response_with_rag(user_input, history, use_rag):
    global model, tokenizer, index, texts
    if use_rag:
        retrieved_texts = retrieve_texts(user_input, index, texts, k=3)
    else:
        retrieved_texts = []
    query = format_query(user_input, retrieved_texts)
    response = generate_response(model, tokenizer, query)
    # 记录对话（openai-style 格式）
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history


# 生成模型回复（原始版本，内部调用）
def generate_response(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(query, "").strip()
    return response


# Web UI，布局类似 ChatGPT，同时增加语音输入、文本转语音及播放切换功能
def launch_web_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 HealthQA Chatbot")
        # 新增开关：调用RAG开关，默认开启
        rag_switch = gr.Checkbox(label="Calling RAG", value=True)
        # 聊天记录
        chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
        # 语音输入组件（支持录音和文件上传）
        audio_input = gr.Audio(type="filepath", label="Voice Input")
        # 文本输入框和发送按钮垂直排列
        with gr.Column():
            user_input = gr.Textbox(placeholder="Please enter your question...")
            send_btn = gr.Button("Send")
        # “语音播报”按钮（不再显示音频播放组件）
        tts_btn = gr.Button("Voice broadcast")
        # 隐藏的音频组件用于自动播放（不会显示在UI中）
        tts_audio = gr.Audio(type="filepath", visible=False)
        # 隐藏状态，用于记录语音是否正在播放，初始值为 False
        play_state = gr.State(False)

        # 录音完成后自动转写并填入文本输入框
        audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=user_input)
        # 发送按钮和回车提交均调用生成回复函数，传入是否调用 RAG 的开关值
        send_btn.click(fn=generate_response_with_rag, inputs=[user_input, chatbot, rag_switch], outputs=chatbot)
        user_input.submit(fn=generate_response_with_rag, inputs=[user_input, chatbot, rag_switch], outputs=chatbot)
        # 点击“语音播报”按钮时，更新隐藏音频组件、按钮文本和播放状态
        tts_btn.click(fn=toggle_tts, inputs=[chatbot, play_state], outputs=[tts_audio, tts_btn, play_state])
    demo.launch(share=True)


# 主函数
def main(model_path=r"C:\Users\zy\Downloads\trained_model_MedQA",
         index_file=r"C:\Users\zy\Downloads\rag\NHS_text_index.faiss",
         texts_file=r"C:\Users\zy\Downloads\rag\NHS_text_texts.json"):
    global model, tokenizer, index, texts
    model, tokenizer = load_model(model_path)
    index, texts = load_text_retrieval_system(index_file, texts_file)
    load_whisper_model()
    launch_web_ui()


# 运行主程序
if __name__ == "__main__":
    main()
