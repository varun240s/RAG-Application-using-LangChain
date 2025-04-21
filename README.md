# RAG-Application-using-LangChain

## While using this RAG application i suggest it is better if you use python3.10 which is a very stable and supporting version.

Some files are not included in this repo 
### 1. create your own .env file and store the given below api keys.
```bash
  GEMINI_API_KEY = 
  GROQ_API_KEY = 
  PINECONE_INDEX_NAME =
  PINECONE_API_KEY = 
  PINECONE_ENVIRONMENT =

```

### 2. create a folder naming pdf_data
   
   * now insert any pdf's data you like , as of me i kept medical data.
   * it is not important to maintain any proper structure in the pdf's
   * if it is a pdf then it is enough.

### 3. create .venv
* run this in the powershell in your existing folder where the entire code is present.
* this is the cd path you should be having
  ../RAG-Application-Using-LangChain
```bash

python -m venv .venv

```
* for mac
```bash
source .venv/bin/activate
```

* for windows
```bash
.venv/bin/activate

```

* for windows in powershell
```bash
.venv\Scripts\Activate.ps1

```

* for windows in CMD
```bash
.venv\Scripts\activate.bat
```

## License

This project is licensed under the Apache 2.0 License.

Note: This project supports Meta’s LLaMA 3 model (`llama3-8b-8192`) which is licensed under Meta’s custom license. You must comply with Meta's terms when using the model. See: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
