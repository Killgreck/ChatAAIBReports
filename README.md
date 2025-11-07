# Sistema RAG para Consulta de Documentos de Accidentes Aéreos

Este proyecto implementa un sistema de Búsqueda y Generación Aumentada (RAG) diseñado para responder preguntas basadas en un conjunto de documentos sobre accidentes aéreos. El sistema utiliza los servicios de Azure OpenAI para la generación de embeddings y texto, y FAISS para la búsqueda vectorial de alta velocidad.

## ¿Cómo funciona el sistema RAG?

Un sistema RAG combina lo mejor de dos mundos: la búsqueda de información y los modelos de lenguaje generativo. En lugar de depender únicamente del conocimiento interno del modelo de lenguaje (que puede estar desactualizado), un sistema RAG primero busca información relevante en una base de datos de documentos y luego utiliza esa información como contexto para generar una respuesta precisa y fundamentada.

El flujo de trabajo es el siguiente:

1.  **Indexación:** Los documentos se dividen en fragmentos de texto más pequeños (chunks), y cada chunk se convierte en un vector numérico (embedding) que captura su significado semántico. Estos embeddings se almacenan en un índice vectorial (FAISS en este caso) para una búsqueda eficiente.
2.  **Búsqueda:** Cuando se realiza una pregunta, esta también se convierte en un embedding. El sistema busca en el índice vectorial los chunks de texto cuyos embeddings sean más similares al embedding de la pregunta.
3.  **Generación:** Los chunks recuperados se combinan para formar un contexto. Este contexto, junto con la pregunta original, se envía a un modelo de lenguaje generativo (como GPT-4), que genera una respuesta coherente y basada en la información proporcionada.

### Justificación de la Configuración

*   **Azure OpenAI:** Se eligió por su robustez, seguridad y escalabilidad, siendo una opción ideal para aplicaciones empresariales.
*   **FAISS (Facebook AI Similarity Search):** Es una biblioteca altamente optimizada para la búsqueda de similitud en vectores. Es extremadamente rápida, lo que es crucial para aplicaciones en tiempo real.
*   **Modelo de Embeddings (`text-embedding-3-small`):** Ofrece un excelente equilibrio entre rendimiento y costo, generando embeddings de alta calidad.
*   **Modelo de Chat (`gpt-4.1-nano`):** Es un modelo potente y eficiente, capaz de generar respuestas coherentes y precisas a partir del contexto proporcionado.

## Configuración del Proyecto

### Prerrequisitos

*   Python 3.8 o superior
*   Una cuenta de Azure con acceso a los servicios de OpenAI

### Instalación

1.  Clona este repositorio:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso del Sistema

Este proyecto se puede ejecutar en tres modos: `index`, `query` y `evaluate`.

### 1. Indexación de Documentos

Antes de poder hacer preguntas, necesitas indexar tus documentos. Asegúrate de que tus archivos (`.pdf`, `.txt`) estén en la carpeta `documentos`.

Para indexar los documentos, ejecuta el siguiente comando. Puedes especificar el modelo de configuración que deseas utilizar (definido en `config.json`).

```bash
python main.py index --model baseline
```

Esto creará dos archivos: `faiss_index.faiss` (el índice vectorial) y `chunks.parquet` (los fragmentos de texto).

### 2. Realizar una Consulta

Una vez que los documentos han sido indexados, puedes hacer preguntas al sistema.

```bash
python main.py query --model baseline --query "how to lock a G-TAWB"
```

El sistema imprimirá la respuesta generada por el modelo de lenguaje.

### 3. Evaluación de Modelos

Este proyecto incluye cuatro modelos preconfigurados en `config.json`:

*   `baseline`: El modelo base.
*   `model_2` y `model_3`: Modelos con diferentes estrategias de chunking.
*   `model_4_mmr`: Un modelo que utiliza Relevancia Marginal Máxima (MMR) para reducir la redundancia en los resultados de búsqueda.

Puedes evaluar el rendimiento de cada modelo utilizando el script `evaluate.py`. Este script utiliza un pequeño conjunto de datos de ejemplo para calcular el `recall@k`.

```bash
python evaluate.py --model baseline
```

Para evaluar otro modelo, simplemente cambia el nombre del modelo:

```bash
python evaluate.py --model model_4_mmr
```
