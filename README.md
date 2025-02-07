# HealthQA_base_MLLM
>This project is a senior course project of Ocean University of China and Heriot-Watt University. It aims to optimize the multimodal large language model in health and support tri-modal input of text, image and voice.

## 🔥 Introduction
The core goals of this project are:
- 🌟 **Support multimodal input**: Combine **text, image, and voice** to improve medical Q&A results.
- 🚀 **Optimize based on LLaVA-Med**: Train and deploy through **AutoDL vGPU-32GB**.
- 🔍 **Dataset**: Contains **text Q&A, image Q&A, and textbooks**, used to build the **RAG database**.

## 📂 Dataset-preparation
### 📥 Data preprocessing
We use Python to unify the data format for subsequent processing. The following are different databases and the formats they contain.
- **NHS dataset** 📜: Contains **Disease, Symptoms and Treatments** ([NHS website](https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/))
- **Text question&answer and textbooks** 📜: Contains **type, question and answer** or **type and text** ([MedQA](https://github.com/jind11/MedQA))
- **Image question&answer** 📜: Contains **type, image, question and answer** ([VQA-Med-2019](https://github.com/abachaa/VQA-Med-2019))

## 🏗 MLLM-model-preparation
### 🌟 Custom knowledge base
Since we need to optimize health-related questions and answers, we choose to use RAG.
- For plain text embedding we choose to use all-MiniLM-L6-v2.([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
- For non-plain text embedding, we choose to use clip-vit-base-patch32.([clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32))
- The specific calling code for this case is:
  [Text_dataset_RAG_construction](directory/RAG_construction/Text_dataset_RAG_construction.ipynb)
  [NHS_dataset_RAG_construction](directory/RAG_construction/NHS_dataset_RAG_construction.ipynb)
  [Image_dataset_RAG_construction](directory/RAG_construction/Image_dataset_RAG_construction.ipynb)

### 📌 MLLM model selection
In order to achieve the tri-modal task goal, we investigated some multi-modal language models and finally decided to choose GLM-4V-9B after comparison.([GLM-4V-9B](https://github.com/THUDM/GLM-4))
- For the specific deployment process, please refer to the official website documentation.
- The calling method of this case:
  [Tri-modal_Before_optimization](directory/final_model/Tri-modal_Before_optimization.ipynb)
  [Trimodal_Optimized](directory/final_model/Trimodal_Optimized.ipynb)

### 🔊 Speech Recognition Model
We chose whisper to complete our speech recognition module.([whisper](https://github.com/openai/whisper))

## 📊 Experimental-Results
### 🎯 Accuracy
We use this metric to evaluate the NHS dataset, input symptoms to see if mllm can correctly return the disease name and countermeasures. We also compare the accuracy before and after adding RAG to determine whether the model has been improved.

### ⏳ Time
We use this metric to evaluate whether adding RAG can improve the response speed of mllm.

### 🔬 Answer Consistency
We compare the results of mllm output with the standard answers on the official website to see whether adding RAG will improve the consistency of answers.
